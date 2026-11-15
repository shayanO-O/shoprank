import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, setup_mlflow, get_paths
from src.features import build_genre_matrix, compute_item_stats, extract_user_histories
from src.metrics import ndcg_at_k, recall_at_k, catalog_coverage, intra_list_diversity, calculate_latency_percentiles
from src.two_tower import UserTower

# Read Git Commit Hash
def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

# Read DVC Data Version Hash
def get_data_version():
    dvc_path = Path("data/raw/ml-25m.zip.dvc")
    if dvc_path.exists():
        with open(dvc_path, "r", encoding="utf-8") as f:
            for line in f:
                if "md5:" in line:
                    return line.split("md5:")[1].strip()
    return "unknown"

# Recommend Candidates From User Representation
def get_retrieval_recs(uid, hist, user_seen, user_tower, index, all_items, item_to_idx, pop_ranked, max_len=50, top_k=200):
    if not hist:
        recs = [m for m in pop_ranked if m not in user_seen][:top_k]
        scores = {m: 0.0 for m in recs}
        return recs, scores
    hist_idx = [item_to_idx[m] for m in hist if m in item_to_idx][-max_len:]
    hist_len = min(len(hist_idx), max_len)
    padded = np.zeros((1, max_len), dtype=np.int64)
    padded[0, :hist_len] = hist_idx
    with torch.no_grad():
        t_hist = torch.tensor(padded, dtype=torch.long)
        t_len = torch.tensor([hist_len], dtype=torch.float32)
        u_vec = user_tower(t_hist, t_len).cpu().numpy()
    dists, indices = index.search(np.ascontiguousarray(u_vec, dtype=np.float32), top_k + len(user_seen))
    recs = []
    scores = {}
    for score, idx in zip(dists[0], indices[0]):
        if idx < 0 or idx >= len(all_items):
            continue
        mid = all_items[idx]
        if mid not in user_seen:
            recs.append(mid)
            scores[mid] = float(score)
            if len(recs) >= top_k:
                break
    return recs, scores

# ReRank Candidates With LightGBM Model
def rank_candidates(candidates, cand_scores, hist, ranker, item_stats, item_genre_dict, num_genres):
    if not candidates:
        return []
    user_genre_vec = np.zeros(num_genres, dtype=np.float32)
    if hist:
        hist_g = [item_genre_dict[m] for m in hist if m in item_genre_dict]
        if hist_g:
            user_genre_vec = np.mean(hist_g, axis=0)
    u_norm = np.linalg.norm(user_genre_vec)
    
    feats = []
    for mid in candidates:
        s = cand_scores.get(mid, 0.0)
        st = item_stats.get(mid, {"popularity": 0.0, "recent_popularity": 0.0, "mean_rating": 3.0, "item_age": 20.0})
        m_g = item_genre_dict.get(mid, np.zeros(num_genres, dtype=np.float32))
        m_norm = np.linalg.norm(m_g)
        overlap = float(np.dot(user_genre_vec, m_g) / (u_norm * m_norm)) if (u_norm > 0 and m_norm > 0) else 0.0
        feats.append([s, st["popularity"], st["recent_popularity"], st["item_age"], st["mean_rating"], float(len(hist)), overlap])
        
    x_val = np.array(feats, dtype=np.float32)
    pred_scores = ranker.predict(x_val)
    ranked_indices = np.argsort(-pred_scores)
    return [candidates[i] for i in ranked_indices]

# Run Full Evaluation On Selected Users
def evaluate_recs(user_list, ground_truth, user_histories, seen_items, user_tower, index, ranker, all_items, item_to_idx, pop_ranked, item_stats, item_genre_dict, num_genres, max_len=50):
    retrieval_recs_all = []
    ranked_recs_all = []
    
    ret_ndcg, ret_recall = [], []
    rank_ndcg, rank_recall = [], []
    
    for uid in user_list:
        actual = ground_truth.get(uid, [])
        if not actual:
            continue
        hist = user_histories.get(uid, [])
        user_seen = seen_items.get(uid, set())
        
        cands, scores = get_retrieval_recs(uid, hist, user_seen, user_tower, index, all_items, item_to_idx, pop_ranked, max_len=max_len, top_k=200)
        retrieval_recs_all.append(cands)
        ret_ndcg.append(ndcg_at_k(actual, cands, k=10))
        ret_recall.append(recall_at_k(actual, cands, k=50))
        
        final_ranked = rank_candidates(cands, scores, hist, ranker, item_stats, item_genre_dict, num_genres)
        ranked_recs_all.append(final_ranked)
        rank_ndcg.append(ndcg_at_k(actual, final_ranked, k=10))
        rank_recall.append(recall_at_k(actual, final_ranked, k=50))
        
    ret_metrics = {
        "ndcg_at_10": float(np.mean(ret_ndcg)) if ret_ndcg else 0.0,
        "recall_at_50": float(np.mean(ret_recall)) if ret_recall else 0.0,
        "catalog_coverage": catalog_coverage(retrieval_recs_all, len(all_items)),
        "diversity": intra_list_diversity(retrieval_recs_all, item_genre_dict)
    }
    
    rank_metrics = {
        "ndcg_at_10": float(np.mean(rank_ndcg)) if rank_ndcg else 0.0,
        "recall_at_50": float(np.mean(rank_recall)) if rank_recall else 0.0,
        "catalog_coverage": catalog_coverage(ranked_recs_all, len(all_items)),
        "diversity": intra_list_diversity(ranked_recs_all, item_genre_dict)
    }
    
    return ret_metrics, rank_metrics

if __name__ == "__main__":
    import mlflow
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    params = load_params()
    setup_mlflow(params, smoke=args.smoke)
    paths = get_paths(params, smoke=args.smoke)
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    
    # Load Training Validation And Test Sets
    train_df = pd.read_parquet(paths["processed"] / "train.parquet")
    val_df = pd.read_parquet(paths["processed"] / "val.parquet")
    test_df = pd.read_parquet(paths["processed"] / "test.parquet")
    items_df = pd.read_parquet(paths["processed"] / "items.parquet")
    
    with open(paths["models"] / "item_id_map.json", "r", encoding="utf-8") as f:
        all_items = json.load(f)
    item_to_idx = {mid: i for i, mid in enumerate(all_items)}
    num_items = len(all_items)
    
    g_matrix, g_map = build_genre_matrix(items_df)
    item_genre_dict = {items_df.iloc[i]["movieId"]: g_matrix[i] for i in range(len(items_df))}
    item_stats = compute_item_stats(train_df, items_df)
    
    # Load Models
    emb_dim = params["retrieval"]["smoke_embedding_dim"] if args.smoke else params["retrieval"]["embedding_dim"]
    max_len = params["retrieval"]["max_history_len"]
    
    user_tower = UserTower(num_items, emb_dim, max_len=max_len)
    state_dict = torch.load(paths["models"] / "two_tower.pt", weights_only=False)
    u_state = {k.replace("user_tower.", ""): v for k, v in state_dict.items() if k.startswith("user_tower.")}
    user_tower.load_state_dict(u_state)
    user_tower.eval()
    
    index = faiss.read_index(str(paths["models"] / "faiss.index"))
    ranker = lgb.Booster(model_file=str(paths["models"] / "ranker.txt"))
    
    user_histories = extract_user_histories(train_df, max_len=max_len)
    seen_items = defaultdict(set)
    for uid, mid in zip(train_df["userId"], train_df["movieId"]):
        seen_items[uid].add(mid)
        
    pop_ranked = train_df[train_df["label"] == 1]["movieId"].value_counts().index.tolist()
    
    # Ground Truth Mappings
    val_pos = val_df[val_df["label"] == 1]
    val_gt = defaultdict(list)
    for uid, mid in zip(val_pos["userId"], val_pos["movieId"]):
        val_gt[uid].append(mid)
        
    test_pos = test_df[test_df["label"] == 1]
    test_gt = defaultdict(list)
    for uid, mid in zip(test_pos["userId"], test_pos["movieId"]):
        test_gt[uid].append(mid)
        
    val_users = list(val_gt.keys())
    test_users = list(test_gt.keys())
    
    if args.smoke:
        val_users = val_users[:200]
        test_users = test_users[:200]
        
    val_cold_users = [u for u in val_users if val_df[val_df["userId"] == u]["is_cold"].iloc[0] == 1]
    test_cold_users = [u for u in test_users if test_df[test_df["userId"] == u]["is_cold"].iloc[0] == 1]
    
    # Compute Validation And Test Metrics
    val_ret_metrics, val_rank_metrics = evaluate_recs(val_users, val_gt, user_histories, seen_items, user_tower, index, ranker, all_items, item_to_idx, pop_ranked, item_stats, item_genre_dict, len(g_map), max_len=max_len)
    test_ret_metrics, test_rank_metrics = evaluate_recs(test_users, test_gt, user_histories, seen_items, user_tower, index, ranker, all_items, item_to_idx, pop_ranked, item_stats, item_genre_dict, len(g_map), max_len=max_len)
    
    _, val_cold_rank_metrics = evaluate_recs(val_cold_users, val_gt, user_histories, seen_items, user_tower, index, ranker, all_items, item_to_idx, pop_ranked, item_stats, item_genre_dict, len(g_map), max_len=max_len)
    _, test_cold_rank_metrics = evaluate_recs(test_cold_users, test_gt, user_histories, seen_items, user_tower, index, ranker, all_items, item_to_idx, pop_ranked, item_stats, item_genre_dict, len(g_map), max_len=max_len)
    
    # Benchmark Pipeline CPU Request Latency
    latency_users = test_users[:params["evaluate"]["latency_samples"]]
    latencies = []
    for uid in latency_users:
        hist = user_histories.get(uid, [])
        user_seen = seen_items.get(uid, set())
        t0 = time.perf_counter()
        cands, scores = get_retrieval_recs(uid, hist, user_seen, user_tower, index, all_items, item_to_idx, pop_ranked, max_len=max_len, top_k=200)
        _ = rank_candidates(cands, scores, hist, ranker, item_stats, item_genre_dict, len(g_map))
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
    latency_stats = calculate_latency_percentiles(latencies)
    
    # Read Baseline Metrics
    base_file = paths["metrics"] / "baselines.json"
    baselines = {}
    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            baselines = json.load(f)
            
    # Assemble Final Results Table
    results = {
        "popular_overall": baselines.get("popular_overall", {}),
        "recent_popular": baselines.get("recent_popular", {}),
        "item_cooccurrence": baselines.get("item_cooccurrence", {}),
        "matrix_factorization": baselines.get("matrix_factorization", {}),
        "retrieval_only": {
            "val": val_ret_metrics,
            "test": test_ret_metrics
        },
        "retrieval_plus_ranker": {
            "val": val_rank_metrics,
            "test": test_rank_metrics,
            "latency": latency_stats
        },
        "cold_start_only": {
            "val": val_cold_rank_metrics,
            "test": test_cold_rank_metrics
        }
    }
    
    # Save Final Metrics File
    res_path = paths["metrics"] / "results.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    # Log Main System Run To MLflow
    git_commit = get_git_commit()
    data_version = get_data_version()
    with mlflow.start_run(run_name="shoprank-retrieval-plus-ranker"):
        mlflow.set_tags({"git_commit": git_commit, "data_version": data_version, "stage": "main_system"})
        mlflow.log_params({
            "embedding_dim": emb_dim,
            "top_k_candidates": 200,
            "ranker_n_estimators": ranker.num_trees()
        })
        mlflow.log_metrics({
            "val_ndcg_at_10": val_rank_metrics["ndcg_at_10"],
            "val_recall_at_50": val_rank_metrics["recall_at_50"],
            "test_ndcg_at_10": test_rank_metrics["ndcg_at_10"],
            "test_recall_at_50": test_rank_metrics["recall_at_50"],
            "latency_p50_ms": latency_stats["p50"],
            "latency_p95_ms": latency_stats["p95"]
        })
        mlflow.log_artifact(str(res_path))
    print("System evaluation and metric logging completed successfully")
