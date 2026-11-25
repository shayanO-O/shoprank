import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
import mlflow

# Load Environment Variables
load_dotenv()
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

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

    exp_name = params.get("project", {}).get("experiment_name", "shoprank-phase2")
    custom_uri = os.environ.get("MLFLOW_TRACKING_URI")
    dagshub_user = os.environ.get("DAGSHUB_USER") or params.get("project", {}).get("dagshub_user")
    dagshub_token = os.environ.get("DAGSHUB_TOKEN")
    repo_name = params.get("project", {}).get("repo_name", "shoprank")

    if custom_uri and "dagshub.com" not in custom_uri:
        tracking_uri = custom_uri
    elif dagshub_user and dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        repo_owner = params.get("project", {}).get("dagshub_user", dagshub_user)
        tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
    elif custom_uri:
        tracking_uri = custom_uri
    else:
        # Default local SQLite tracking if no valid remote tracking credentials
        db_path = Path("mlruns.db").resolve()
        tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e:
        # Gracefully handle remote auth/403 or network failure by falling back to local SQLite store
        import logging
        logging.getLogger(__name__).warning(
            f"Failed to initialize remote MLflow tracking at {tracking_uri} ({e}). "
            f"Falling back to local SQLite tracking store."
        )
        db_path = Path("mlruns.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")
        mlflow.set_experiment(exp_name)

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
