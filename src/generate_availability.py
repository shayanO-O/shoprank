import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

def generate_availability(items_path=None, output_path=None, unavailable_ratio=0.02, seed=42):
    """
    Generates data/processed/availability.parquet by flagging ~2% of movies as unavailable.
    """
    rng = np.random.default_rng(seed)
    items_p = Path(items_path) if items_path else Path("data/processed/items.parquet")
    out_p = Path(output_path) if output_path else Path("data/processed/availability.parquet")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    if items_p.exists():
        df_items = pd.read_parquet(items_p)
        movie_ids = df_items["movieId"].unique()
    else:
        # Fallback to item_id_map.json if parquet is not materialized yet
        map_path = Path("models/item_id_map.json")
        if map_path.exists():
            with open(map_path, "r", encoding="utf-8") as f:
                movie_ids = np.array(json.load(f), dtype=np.int64)
        else:
            # Synthetic fallback for tests
            movie_ids = np.arange(1, 1001, dtype=np.int64)
            
    n_items = len(movie_ids)
    n_unavailable = max(1, int(n_items * unavailable_ratio))
    unavailable_indices = rng.choice(n_items, size=n_unavailable, replace=False)
    
    is_avail = np.ones(n_items, dtype=bool)
    is_avail[unavailable_indices] = False
    
    df_avail = pd.DataFrame({
        "movieId": movie_ids,
        "is_available": is_avail
    })
    
    df_avail.to_parquet(out_p, index=False)
    print(f"Generated {out_p}: {n_unavailable}/{n_items} items marked unavailable ({unavailable_ratio * 100:.1f}%)")
    return df_avail

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate movie catalog availability status")
    parser.add_argument("--items-path", type=str, default="data/processed/items.parquet", help="Path to items parquet")
    parser.add_argument("--output-path", type=str, default="data/processed/availability.parquet", help="Output path")
    parser.add_argument("--ratio", type=float, default=0.02, help="Ratio of unavailable items")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    generate_availability(
        items_path=args.items_path,
        output_path=args.output_path,
        unavailable_ratio=args.ratio,
        seed=args.seed
    )
