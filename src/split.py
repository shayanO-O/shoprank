import argparse
import sys
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

# Read Processed Interactions
df = pd.read_parquet(paths["processed"] / "interactions.parquet")

# Compute Timestamp Quantiles
q_train = params["split"]["train_quantile"]
q_ranker = params["split"]["ranker_quantile"]
q_val = params["split"]["val_quantile"]

t_train = df["timestamp"].quantile(q_train)
t_ranker = df["timestamp"].quantile(q_ranker)
t_val = df["timestamp"].quantile(q_val)

# Split Dataset By Time Windows
df_train = df[df["timestamp"] <= t_train].copy()
df_ranker = df[(df["timestamp"] > t_train) & (df["timestamp"] <= t_ranker)].copy()
df_val = df[(df["timestamp"] > t_ranker) & (df["timestamp"] <= t_val)].copy()
df_test = df[df["timestamp"] > t_val].copy()

# Identify Cold Start Users
train_users = set(df_train["userId"].unique())
df_val["is_cold"] = (~df_val["userId"].isin(train_users)).astype(int)
df_test["is_cold"] = (~df_test["userId"].isin(train_users)).astype(int)

# Save Window Datasets
df_train.to_parquet(paths["processed"] / "train.parquet", index=False)
df_ranker.to_parquet(paths["processed"] / "ranker.parquet", index=False)
df_val.to_parquet(paths["processed"] / "val.parquet", index=False)
df_test.to_parquet(paths["processed"] / "test.parquet", index=False)
