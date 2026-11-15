import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
import pandas as pd

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, get_paths

# Parse Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()

# Load Configuration Parameters
params = load_params()
paths = get_paths(params, smoke=args.smoke)
paths["metrics"].mkdir(parents=True, exist_ok=True)

# Read Raw Datasets From Zip
zip_path = params["data"]["raw_path"]
nrows = params["data"]["smoke_max_rows"] if args.smoke else None

with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("ml-25m/ratings.csv") as f_ratings:
        df_ratings = pd.read_csv(f_ratings, nrows=nrows)
    with z.open("ml-25m/movies.csv") as f_movies:
        df_movies = pd.read_csv(f_movies)

# Execute Data Validation Checks
duplicate_count = int(df_ratings.duplicated(subset=["userId", "movieId"]).sum())
current_time = time.time()
future_count = int((df_ratings["timestamp"] > current_time).sum())
valid_movie_ids = set(df_movies["movieId"].unique())
missing_item_count = int((~df_ratings["movieId"].isin(valid_movie_ids)).sum())
null_key_count = int(df_ratings[["userId", "movieId", "rating", "timestamp"]].isnull().sum().sum())

results = {
    "total_ratings_checked": int(len(df_ratings)),
    "duplicate_events": duplicate_count,
    "future_timestamps": future_count,
    "missing_item_ids": missing_item_count,
    "null_key_columns": null_key_count,
    "passed": bool(duplicate_count == 0 and future_count == 0 and missing_item_count == 0 and null_key_count == 0)
}

# Save Validation Metrics
out_file = paths["metrics"] / "validation.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

if not results["passed"]:
    sys.exit("Data validation failed")
