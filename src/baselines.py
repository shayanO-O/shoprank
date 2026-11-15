import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import mlflow

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, setup_mlflow, get_paths
from src.features import build_genre_matrix, extract_user_histories
from src.metrics import ndcg_at_k, recall_at_k, catalog_coverage, intra_list_diversity

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

# Run Common Baseline Evaluation Loop
def evaluate_baseline(recommender_fn, val_users, user_ground_truth, seen_items, all_items, item_genre_dict):
    all_top_recs = []
    ndcg_scores = []
    recall_scores = []
    
    for uid in val_users:
        actual = user_ground_truth.get(uid, [])
        if not actual:
            continue
        user_seen = seen_items.get(uid, set())
        recs = recommender_fn(uid, user_seen)
        all_top_recs.append(recs)
        ndcg_scores.append(ndcg_at_k(actual, recs, k=10))
        recall_scores.append(recall_at_k(actual, recs, k=50))
        
    return {
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "recall_at_50": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "catalog_coverage": catalog_coverage(all_top_recs, len(all_items)),
        "diversity": intra_list_diversity(all_top_recs, item_genre_dict)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    params = load_params()
    setup_mlflow(params, smoke=args.smoke)
    paths = get_paths(params, smoke=args.smoke)
    paths["metrics"].mkdir(parents=True, exist_ok=True)
    
    # Load Training And Validation Sets
    train_df = pd.read_parquet(paths["processed"] / "train.parquet")
    val_df = pd.read_parquet(paths["processed"] / "val.parquet")
    items_df = pd.read_parquet(paths["processed"] / "items.parquet")
    
    all_items = items_df["movieId"].unique().tolist()
    g_matrix, g_map = build_genre_matrix(items_df)
    item_genre_dict = {items_df.iloc[i]["movieId"]: g_matrix[i] for i in range(len(items_df))}
    
    # Extract Ground Truth And Seen Sets
    seen_items = defaultdict(set)
    for uid, mid in zip(train_df["userId"], train_df["movieId"]):
        seen_items[uid].add(mid)
        
    val_positives = val_df[val_df["label"] == 1]
    user_ground_truth = defaultdict(list)
    for uid, mid in zip(val_positives["userId"], val_positives["movieId"]):
        user_ground_truth[uid].append(mid)
    val_users = list(user_ground_truth.keys())
    
    git_commit = get_git_commit()
    data_version = get_data_version()
    baseline_results = {}

    # Train And Evaluate Most Popular Overall
    pop_ranked = train_df[train_df["label"] == 1]["movieId"].value_counts().index.tolist()
    def rec_popular(uid, user_seen):
        return [m for m in pop_ranked if m not in user_seen][:50]
        
    res_pop = evaluate_baseline(rec_popular, val_users, user_ground_truth, seen_items, all_items, item_genre_dict)
    baseline_results["popular_overall"] = res_pop
    with mlflow.start_run(run_name="baseline-most-popular"):
        mlflow.set_tags({"git_commit": git_commit, "data_version": data_version, "baseline": "most_popular"})
        mlflow.log_metrics(res_pop)

    # Train And Evaluate Recent Popular Baseline
    max_ts = train_df["timestamp"].max() if len(train_df) > 0 else 0
    recent_train = train_df[(train_df["timestamp"] >= max_ts - (30 * 24 * 3600)) & (train_df["label"] == 1)]
    recent_ranked = recent_train["movieId"].value_counts().index.tolist()
    fallback_ranked = pop_ranked
    def rec_recent(uid, user_seen):
        recs = [m for m in recent_ranked if m not in user_seen]
        if len(recs) < 50:
            for m in fallback_ranked:
                if m not in user_seen and m not in recs:
                    recs.append(m)
                if len(recs) >= 50:
                    break
        return recs[:50]
        
    res_recent = evaluate_baseline(rec_recent, val_users, user_ground_truth, seen_items, all_items, item_genre_dict)
    baseline_results["recent_popular"] = res_recent
    with mlflow.start_run(run_name="baseline-recent-popular"):
        mlflow.set_tags({"git_commit": git_commit, "data_version": data_version, "baseline": "recent_popular"})
        mlflow.log_metrics(res_recent)

    # Train And Evaluate Item Cooccurrence Baseline
    train_positives = train_df[train_df["label"] == 1]
    user_histories = extract_user_histories(train_positives, max_len=50)
    cooccur = defaultdict(lambda: defaultdict(int))
    for hist in user_histories.values():
        for i in range(len(hist)):
            for j in range(i + 1, min(i + 10, len(hist))):
                cooccur[hist[i]][hist[j]] += 1
                cooccur[hist[j]][hist[i]] += 1
                
    def rec_cooccur(uid, user_seen):
        hist = user_histories.get(uid, [])
        if not hist:
            return rec_recent(uid, user_seen)
        scores = defaultdict(int)
        for mid in hist:
            for cand, count in cooccur[mid].items():
                if cand not in user_seen:
                    scores[cand] += count
        ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        if len(ranked) < 50:
            for m in rec_recent(uid, user_seen):
                if m not in ranked:
                    ranked.append(m)
                if len(ranked) >= 50:
                    break
        return ranked[:50]
        
    res_cooccur = evaluate_baseline(rec_cooccur, val_users, user_ground_truth, seen_items, all_items, item_genre_dict)
    baseline_results["item_cooccurrence"] = res_cooccur
    with mlflow.start_run(run_name="baseline-item-cooccurrence"):
        mlflow.set_tags({"git_commit": git_commit, "data_version": data_version, "baseline": "item_cooccurrence"})
        mlflow.log_metrics(res_cooccur)

    # Train And Evaluate Matrix Factorization Baseline
    unique_train_users = sorted(train_positives["userId"].unique().tolist())
    u2idx = {u: i for i, u in enumerate(unique_train_users)}
    i2idx = {m: i for i, m in enumerate(all_items)}
    
    rows = [u2idx[u] for u in train_positives["userId"] if u in u2idx and train_positives.iloc[0]["movieId"] in i2idx]
    rows = []
    cols = []
    for u, m in zip(train_positives["userId"], train_positives["movieId"]):
        if u in u2idx and m in i2idx:
            rows.append(u2idx[u])
            cols.append(i2idx[m])
    data = np.ones(len(rows), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(len(unique_train_users), len(all_items)))
    
    n_comp = 16 if args.smoke else 64
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    user_factors = svd.fit_transform(mat)
    item_factors = svd.components_.T
    
    def rec_svd(uid, user_seen):
        if uid not in u2idx:
            return rec_recent(uid, user_seen)
        u_idx = u2idx[uid]
        u_vec = user_factors[u_idx]
        scores = np.dot(item_factors, u_vec)
        top_indices = np.argsort(-scores)
        recs = []
        for idx in top_indices:
            mid = all_items[idx]
            if mid not in user_seen:
                recs.append(mid)
            if len(recs) >= 50:
                break
        return recs
        
    res_svd = evaluate_baseline(rec_svd, val_users, user_ground_truth, seen_items, all_items, item_genre_dict)
    baseline_results["matrix_factorization"] = res_svd
    with mlflow.start_run(run_name="baseline-matrix-factorization"):
        mlflow.set_tags({"git_commit": git_commit, "data_version": data_version, "baseline": "matrix_factorization"})
        mlflow.log_metrics(res_svd)

    # Save Baseline Evaluation Metrics
    with open(paths["metrics"] / "baselines.json", "w", encoding="utf-8") as f:
        json.dump(baseline_results, f, indent=2)
    print("Baseline evaluation completed successfully")
