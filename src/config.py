import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import mlflow

# Load Environment Variables
load_dotenv()

# Read Configuration File
def load_params(config_path="params.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Configure MLflow Tracking
def setup_mlflow(params, smoke=False):
    if smoke:
        smoke_dir = Path("scratch/smoke/mlruns")
        smoke_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(smoke_dir.resolve().as_uri())
        mlflow.set_experiment("smoke-test")
        return
    dagshub_user = os.environ.get("DAGSHUB_USER") or params["project"]["dagshub_user"]
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    repo_name = params["project"]["repo_name"]
    if dagshub_user and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    tracking_uri = f"https://dagshub.com/{dagshub_user}/{repo_name}.mlflow"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["project"]["experiment_name"])

# Resolve Output Directory Paths
def get_paths(params, smoke=False):
    if smoke:
        base = Path("scratch/smoke")
        return {
            "processed": base / "processed",
            "models": base / "models",
            "metrics": base / "metrics"
        }
    return {
        "processed": Path(params["data"]["processed_dir"]),
        "models": Path("models"),
        "metrics": Path("metrics")
    }
