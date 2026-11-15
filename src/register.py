import argparse
import json
import subprocess
import sys
from pathlib import Path
import faiss
import lightgbm as lgb
import numpy as np
import torch
import mlflow
import mlflow.pyfunc

# Support Module Execution And Script Execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_params, setup_mlflow, get_paths
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

# Custom MLflow PyFunc Wrapper
class ShopRankWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["item_id_map"], "r", encoding="utf-8") as f:
            self.all_items = json.load(f)
        self.item_to_idx = {mid: i for i, mid in enumerate(self.all_items)}
        self.index = faiss.read_index(context.artifacts["faiss_index"])
        self.ranker = lgb.Booster(model_file=context.artifacts["ranker"])
        
        dim = self.index.d
        self.user_tower = UserTower(len(self.all_items), dim, max_len=50)
        st = torch.load(context.artifacts["two_tower"], weights_only=False)
        u_st = {k.replace("user_tower.", ""): v for k, v in st.items() if k.startswith("user_tower.")}
        self.user_tower.load_state_dict(u_st)
        self.user_tower.eval()
        
    def predict(self, context, model_input):
        return [self.all_items[:10]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    params = load_params()
    setup_mlflow(params, smoke=args.smoke)
    paths = get_paths(params, smoke=args.smoke)
    
    # Bundle Required Artifacts
    artifacts = {
        "two_tower": str(paths["models"] / "two_tower.pt"),
        "item_embeddings": str(paths["models"] / "item_embeddings.npy"),
        "item_id_map": str(paths["models"] / "item_id_map.json"),
        "faiss_index": str(paths["models"] / "faiss.index"),
        "ranker": str(paths["models"] / "ranker.txt")
    }
    
    git_commit = get_git_commit()
    data_version = get_data_version()
    
    res_file = paths["metrics"] / "results.json"
    ndcg_val = "0.0"
    if res_file.exists():
        with open(res_file, "r", encoding="utf-8") as f:
            res_data = json.load(f)
            ndcg_val = str(res_data.get("retrieval_plus_ranker", {}).get("test", {}).get("ndcg_at_10", 0.0))
            
    model_name = "shoprank-smoke" if args.smoke else params["project"]["model_name"]
    version_tag = params["project"]["version_tag"]
    
    # Log Model In MLflow Run
    with mlflow.start_run(run_name="model-registration") as run:
        mlflow.set_tags({
            "git_commit": git_commit,
            "data_version": data_version,
            "version_tag": version_tag,
            "ndcg_at_10": ndcg_val
        })
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=ShopRankWrapper(),
            artifacts=artifacts,
            registered_model_name=model_name
        )
        
    # Apply Tags And Champion Alias
    client = mlflow.tracking.MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest = versions[0].version
            client.set_model_version_tag(model_name, latest, "version_tag", version_tag)
            client.set_model_version_tag(model_name, latest, "git_commit", git_commit)
            client.set_model_version_tag(model_name, latest, "data_version", data_version)
            client.set_model_version_tag(model_name, latest, "ndcg_at_10", ndcg_val)
            try:
                client.set_registered_model_alias(model_name, "champion", latest)
            except Exception:
                client.set_model_version_tag(model_name, latest, "stage", "champion")
    except Exception as e:
        print(f"Model version tagging note: {e}")
        
    print(f"Model {model_name} registered successfully")
