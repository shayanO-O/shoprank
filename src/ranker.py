import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, get_paths
from src.features import build_genre_matrix, compute_item_stats, extract_user_histories
from src.two_tower import UserTower

# Generate Candidates Using FAISS Index
def retrieve_user_candidates(user_vec, index, all_items, seen_set, top_k=200):
    scores, indices = index.search(np.ascontiguousarray(user_vec, dtype=np.float32), top_k + len(seen_set))
    candidates = []
    cand_scores = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(all_items):
            continue
        mid = all_items[idx]
        if mid not in seen_set:
            candidates.append(mid)
            cand_scores[mid] = float(score)
            if len(candidates) >= top_k:
                break
    return candidates, cand_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    params = load_params()
    paths = get_paths(params, smoke=args.smoke)
    paths["models"].mkdir(parents=True, exist_ok=True)
    
    # Load Processed Datasets
    train_df = pd.read_parquet(paths["processed"] / "train.parquet")
    ranker_df = pd.read_parquet(paths["processed"] / "ranker.parquet")
    items_df = pd.read_parquet(paths["processed"] / "items.parquet")
    
    with open(paths["models"] / "item_id_map.json", "r", encoding="utf-8") as f:
        all_items = json.load(f)
    item_to_idx = {mid: i for i, mid in enumerate(all_items)}
    num_items = len(all_items)
    
    # Load Index And Model Components
    index = faiss.read_index(str(paths["models"] / "faiss.index"))
    g_matrix, g_map = build_genre_matrix(items_df)
    item_genre_dict = {items_df.iloc[i]["movieId"]: g_matrix[i] for i in range(len(items_df))}
    item_stats = compute_item_stats(train_df, items_df)
    
    emb_dim = params["retrieval"]["smoke_embedding_dim"] if args.smoke else params["retrieval"]["embedding_dim"]
    max_len = params["retrieval"]["max_history_len"]
    
    user_tower = UserTower(num_items, emb_dim, max_len=max_len)
    state_dict = torch.load(paths["models"] / "two_tower.pt", weights_only=False)
    u_state = {k.replace("user_tower.", ""): v for k, v in state_dict.items() if k.startswith("user_tower.")}
    user_tower.load_state_dict(u_state)
    user_tower.eval()
    
    # Prepare User Historical Profiles
    user_histories = extract_user_histories(train_df, max_len=max_len)
    seen_items = defaultdict(set)
    for uid, mid in zip(train_df["userId"], train_df["movieId"]):
        seen_items[uid].add(mid)
        
    ranker_positives = ranker_df[ranker_df["label"] == 1]
    user_ranker_positives = defaultdict(set)
    for uid, mid in zip(ranker_positives["userId"], ranker_positives["movieId"]):
        user_ranker_positives[uid].add(mid)
        
    pop_ranked = train_df[train_df["label"] == 1]["movieId"].value_counts().index.tolist()
    
    # Extract Feature Vectors For Ranker Training
    feature_rows = []
    labels = []
    groups = []
    
    ranker_users = sorted(ranker_df["userId"].unique().tolist())
    if args.smoke:
        ranker_users = ranker_users[:200]
        
    with torch.no_grad():
        for uid in ranker_users:
            hist = user_histories.get(uid, [])
            user_seen = seen_items.get(uid, set())
            
            if hist:
                hist_idx = [item_to_idx[m] for m in hist if m in item_to_idx][-max_len:]
                hist_len = min(len(hist_idx), max_len)
                padded_hist = np.zeros((1, max_len), dtype=np.int64)
                padded_hist[0, :hist_len] = hist_idx
                t_hist = torch.tensor(padded_hist, dtype=torch.long)
                t_len = torch.tensor([hist_len], dtype=torch.float32)
                u_vec = user_tower(t_hist, t_len).cpu().numpy()
                candidates, cand_scores = retrieve_user_candidates(u_vec, index, all_items, user_seen, top_k=200)
            else:
                candidates = [m for m in pop_ranked if m not in user_seen][:200]
                cand_scores = {m: 0.0 for m in candidates}
                hist_len = 0
                
            if not candidates:
                continue
                
            user_genre_vec = np.zeros(len(g_map), dtype=np.float32)
            if hist:
                hist_g = [item_genre_dict[m] for m in hist if m in item_genre_dict]
                if hist_g:
                    user_genre_vec = np.mean(hist_g, axis=0)
            u_g_norm = np.linalg.norm(user_genre_vec)
            
            group_count = 0
            user_positives = user_ranker_positives.get(uid, set())
            
            for mid in candidates:
                retrieval_score = cand_scores.get(mid, 0.0)
                stats = item_stats.get(mid, {"popularity": np.nan, "recent_popularity": np.nan, "mean_rating": 3.0, "item_age": 20.0})
                
                mid_g = item_genre_dict.get(mid, np.zeros(len(g_map), dtype=np.float32))
                m_g_norm = np.linalg.norm(mid_g)
                overlap = float(np.dot(user_genre_vec, mid_g) / (u_g_norm * m_g_norm)) if (u_g_norm > 0 and m_g_norm > 0) else 0.0
                
                # 10% popularity dropout during training so zero-history items are not buried
                pop_val = stats.get("popularity", np.nan)
                rec_pop_val = stats.get("recent_popularity", np.nan)
                if np.random.rand() < 0.10:
                    pop_val = np.nan
                    rec_pop_val = np.nan
                
                feat = [
                    retrieval_score,
                    pop_val,
                    rec_pop_val,
                    stats.get("item_age", 20.0),
                    stats.get("mean_rating", 3.0),
                    float(hist_len),
                    overlap
                ]
                label = 1 if mid in user_positives else 0
                feature_rows.append(feat)
                labels.append(label)
                group_count += 1
                
            if group_count > 0:
                groups.append(group_count)

    # Train LightGBM LambdaRank Model
    x_train = np.array(feature_rows, dtype=np.float32)
    y_train = np.array(labels, dtype=np.int32)
    
    n_est = params["ranker"]["smoke_n_estimators"] if args.smoke else params["ranker"]["n_estimators"]
    lr = params["ranker"]["learning_rate"]
    num_leaves = params["ranker"]["num_leaves"]
    
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_est,
        learning_rate=lr,
        num_leaves=num_leaves,
        random_state=42
    )
    ranker.fit(x_train, y_train, group=groups)
    
    # Save Trained Ranker Model
    ranker_out = paths["models"] / "ranker.txt"
    ranker.booster_.save_model(str(ranker_out))
    print(f"Ranker trained successfully on {len(groups)} groups and {len(x_train)} samples")
