import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, List
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

# Calculate SHA256 Checksum For File
def calculate_file_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

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
        manifest_path = context.artifacts.get("manifest")
        if manifest_path and Path(manifest_path).exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            for artifact_name, expected_sha in manifest.get("artifacts", {}).items():
                target_path = context.artifacts.get(artifact_name)
                if target_path and Path(target_path).exists():
                    actual_sha = calculate_file_sha256(target_path)
                    if actual_sha != expected_sha:
                        raise RuntimeError(
                            f"VersionMismatchError: SHA256 mismatch for {artifact_name} "
                            f"(expected {expected_sha}, got {actual_sha})"
                        )

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
        
    def predict(self, context: Any, model_input: List[int]) -> List[List[int]]:
        return [self.all_items[:10]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    params = load_params()
    setup_mlflow(params, smoke=args.smoke)
    paths = get_paths(params, smoke=args.smoke)
    paths["models"].mkdir(parents=True, exist_ok=True)
    
    # Compute Artifact Checksums & Bundle Manifest
    artifact_files = {
        "two_tower": paths["models"] / "two_tower.pt",
        "item_embeddings": paths["models"] / "item_embeddings.npy",
        "item_id_map": paths["models"] / "item_id_map.json",
        "faiss_index": paths["models"] / "faiss.index",
        "ranker": paths["models"] / "ranker.txt"
    }
    
    artifact_hashes = {}
    for name, p in artifact_files.items():
        if p.exists():
            h = calculate_file_sha256(p)
            artifact_hashes[name] = h
            artifact_hashes[p.name] = h
        else:
            artifact_hashes[name] = "missing"
            artifact_hashes[p.name] = "missing"
            
    concat_hashes = "".join([artifact_hashes[k] for k in sorted(artifact_hashes.keys())])
    bundle_hash = hashlib.sha256(concat_hashes.encode("utf-8")).hexdigest()
    
    git_commit = get_git_commit()
    data_version = get_data_version()
    version_tag = params["project"]["version_tag"]
    
    res_file = paths["metrics"] / "results.json"
    ndcg_val = "0.0"
    if res_file.exists():
        with open(res_file, "r", encoding="utf-8") as f:
            res_data = json.load(f)
            ndcg_val = str(res_data.get("retrieval_plus_ranker", {}).get("test", {}).get("ndcg_at_10", 0.0))
            
    manifest_data = {
        "bundle_name": "shoprank-bundle",
        "bundle_hash": bundle_hash,
        "version_tag": version_tag,
        "git_commit": git_commit,
        "dvc_data_version": data_version,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "artifacts": artifact_hashes,
        "metrics": {
            "ndcg_at_10": float(ndcg_val)
        }
    }
    
    manifest_path = paths["models"] / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"Generated bundle manifest: {manifest_path} (bundle_hash: {bundle_hash[:12]}...)")

    # Bundle Required Artifacts
    artifacts = {
        "manifest": str(manifest_path),
        "two_tower": str(artifact_files["two_tower"]),
        "item_embeddings": str(artifact_files["item_embeddings"]),
        "item_id_map": str(artifact_files["item_id_map"]),
        "faiss_index": str(artifact_files["faiss_index"]),
        "ranker": str(artifact_files["ranker"])
    }
    
    model_name = "shoprank-bundle-smoke" if args.smoke else "shoprank-bundle"
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
