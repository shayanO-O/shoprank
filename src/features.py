import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, get_paths

# Build Genre Multi Hot Representation
def build_genre_matrix(items_df):
    genres_set = set()
    for g_str in items_df["genres"].dropna():
        for g in str(g_str).split("|"):
            if g and g != "(no genres listed)":
                genres_set.add(g)
    genre_list = sorted(list(genres_set))
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}
    num_genres = len(genre_list)
    
    matrix = np.zeros((len(items_df), num_genres), dtype=np.float32)
    for row_idx, g_str in enumerate(items_df["genres"]):
        for g in str(g_str).split("|"):
            if g in genre_to_idx:
                matrix[row_idx, genre_to_idx[g]] = 1.0
    return matrix, genre_to_idx

# Compute Historical Item Statistics
def compute_item_stats(train_df, items_df):
    item_counts = train_df["movieId"].value_counts().to_dict()
    mean_ratings = train_df.groupby("movieId")["rating"].mean().to_dict()
    
    max_ts = train_df["timestamp"].max() if len(train_df) > 0 else 0
    recent_cutoff = max_ts - (30 * 24 * 3600)
    recent_df = train_df[train_df["timestamp"] >= recent_cutoff]
    recent_counts = recent_df["movieId"].value_counts().to_dict()
    
    stats = {}
    current_year = 2020
    for _, row in items_df.iterrows():
        mid = int(row["movieId"])
        stats[mid] = {
            "popularity": float(item_counts.get(mid, 0.0)),
            "recent_popularity": float(recent_counts.get(mid, 0.0)),
            "mean_rating": float(mean_ratings.get(mid, 3.0)),
            "item_age": float(max(0, current_year - int(row.get("year", 2000))))
        }
    return stats

# Extract Historical User Sequences
def extract_user_histories(interactions_df, max_len=50):
    positives = interactions_df[interactions_df["label"] == 1].sort_values("timestamp")
    user_histories = {}
    for uid, group in positives.groupby("userId"):
        user_histories[int(uid)] = group["movieId"].tolist()[-max_len:]
    return user_histories

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    params = load_params()
    paths = get_paths(params, smoke=args.smoke)
    
    # Verify Feature Pipeline Components
    items_df = pd.read_parquet(paths["processed"] / "items.parquet")
    train_df = pd.read_parquet(paths["processed"] / "train.parquet")
    g_matrix, g_map = build_genre_matrix(items_df)
    item_stats = compute_item_stats(train_df, items_df)
    user_hist = extract_user_histories(train_df)
    print(f"Features verified: {len(g_map)} genres, {len(item_stats)} items, {len(user_hist)} users")
