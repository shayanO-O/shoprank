import argparse
import re
import sys
import zipfile
from pathlib import Path
import numpy as np
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
paths["processed"].mkdir(parents=True, exist_ok=True)

# Read Raw Datasets From Zip
zip_path = params["data"]["raw_path"]
nrows = params["data"]["smoke_max_rows"] if args.smoke else None

with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("ml-25m/ratings.csv") as f_ratings:
        df_ratings = pd.read_csv(f_ratings, nrows=nrows)
    with z.open("ml-25m/movies.csv") as f_movies:
        df_movies = pd.read_csv(f_movies)

# Filter Users And Items By Interaction Frequency
min_user_ratings = 5 if args.smoke else params["data"]["min_user_ratings"]
min_item_ratings = 5 if args.smoke else params["data"]["min_item_ratings"]
sample_users = 2000 if args.smoke else params["data"]["sample_users"]
seed = params["data"]["seed"]

user_counts = df_ratings["userId"].value_counts()
eligible_users = user_counts[user_counts >= min_user_ratings].index.to_numpy()

if len(eligible_users) > sample_users:
    rng = np.random.default_rng(seed)
    selected_users = rng.choice(eligible_users, size=sample_users, replace=False)
else:
    selected_users = eligible_users

df_filtered = df_ratings[df_ratings["userId"].isin(set(selected_users))].copy()

item_counts = df_filtered["movieId"].value_counts()
eligible_items = item_counts[item_counts >= min_item_ratings].index.to_numpy()
df_filtered = df_filtered[df_filtered["movieId"].isin(set(eligible_items))].copy()

# Add Binary Positive Interaction Label
df_filtered["label"] = (df_filtered["rating"] >= 4.0).astype(int)

# Extract Movie Features
def parse_year(title):
    match = re.search(r"\((\d{4})\)", str(title))
    return int(match.group(1)) if match else 2000

df_movies["year"] = df_movies["title"].apply(parse_year)
df_items = df_movies[df_movies["movieId"].isin(set(eligible_items))].copy()

# Save Processed Parquet Datasets
df_filtered.to_parquet(paths["processed"] / "interactions.parquet", index=False)
df_items.to_parquet(paths["processed"] / "items.parquet", index=False)
