import json
from pathlib import Path
import faiss
import lightgbm as lgb
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.register import calculate_file_sha256
from src.serving.app import ShopRankServingEngine, create_app
from src.two_tower import UserTower

@pytest.fixture(scope="module")
def mock_model_bundle(tmp_path_factory):
    bundle_dir = tmp_path_factory.mktemp("mock_bundle")
    num_items = 30
    emb_dim = 16
    items = list(range(1001, 1001 + num_items))
    
    # 1. item_id_map.json
    map_path = bundle_dir / "item_id_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(items, f)
        
    # 2. item_embeddings.npy
    embs = np.random.randn(num_items, emb_dim).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    emb_path = bundle_dir / "item_embeddings.npy"
    np.save(emb_path, embs)
    
    # 3. faiss.index
    index = faiss.IndexFlatIP(emb_dim)
    index.add(embs)
    index_path = bundle_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    
    # 4. two_tower.pt
    tower = UserTower(num_items=num_items, emb_dim=emb_dim, max_len=50)
    st = {f"user_tower.{k}": v for k, v in tower.state_dict().items()}
    tt_path = bundle_dir / "two_tower.pt"
    torch.save(st, tt_path)
    
    # 5. ranker.txt (Train dummy LightGBM model on 7 features)
    X = np.random.randn(50, 7).astype(np.float32)
    y = np.random.randint(0, 2, size=50)
    train_data = lgb.Dataset(X, label=y)
    bst = lgb.train({"objective": "binary", "verbosity": -1, "min_data_in_leaf": 2}, train_data, num_boost_round=5)
    ranker_path = bundle_dir / "ranker.txt"
    bst.save_model(str(ranker_path))
    
    # 6. manifest.json
    artifacts = {
        "item_id_map.json": calculate_file_sha256(map_path),
        "item_embeddings.npy": calculate_file_sha256(emb_path),
        "faiss.index": calculate_file_sha256(index_path),
        "two_tower.pt": calculate_file_sha256(tt_path),
        "ranker.txt": calculate_file_sha256(ranker_path)
    }
    manifest = {
        "model_name": "shoprank-mock",
        "version_tag": "v0.1.0-test",
        "bundle_hash": "mock_bundle_hash_12345",
        "git_commit": "testcommit",
        "data_version": "testdvc",
        "artifacts": artifacts
    }
    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return bundle_dir

@pytest.fixture(scope="module")
def client(mock_model_bundle):
    engine = ShopRankServingEngine(
        model_dir=mock_model_bundle,
        redis_url="disabled",
        model_version="v0.1.0-test"
    )
    engine.load_bundle()
    app = create_app(engine=engine)
    return TestClient(app)

def test_healthz_and_ready(client):
    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "alive"
    
    res_ready = client.get("/ready")
    assert res_ready.status_code == 200
    assert res_ready.json()["status"] == "ready"

def test_version_endpoint(client):
    res = client.get("/version")
    assert res.status_code == 200
    data = res.json()
    assert data["model_version"] == "v0.1.0-test"
    assert data["bundle_hash"] == "mock_bundle_hash_12345"
    assert "two_tower.pt" in data["artifacts"]

def test_recommend_endpoint_cold_user(client):
    payload = {
        "user_id": 9999,
        "top_k": 5
    }
    res = client.post("/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == 9999
    assert len(data["recommendations"]) == 5
    assert data["model_version"] == "v0.1.0-test"
    assert data["latency_ms"] > 0.0
    
    # Check ranking order
    ranks = [r["rank"] for r in data["recommendations"]]
    assert ranks == [1, 2, 3, 4, 5]

def test_recommend_with_as_of(client):
    payload = {
        "user_id": 101,
        "top_k": 3,
        "as_of": 1700000000.0
    }
    res = client.post("/recommend", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["recommendations"]) == 3

def test_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    content = res.text
    assert "shoprank_recommendation_requests_total" in content
    assert "shoprank_recommendation_latency_seconds" in content

def test_tampered_manifest_checksum_rejection(tmp_path):
    # Corrupt an artifact and check that load_bundle raises ValueError
    bad_dir = tmp_path / "bad_bundle"
    bad_dir.mkdir()
    f1 = bad_dir / "item_id_map.json"
    f1.write_text("[1, 2, 3]")
    manifest = {
        "bundle_hash": "bad",
        "artifacts": {
            "item_id_map.json": "0000000000000000000000000000000000000000000000000000000000000000"
        }
    }
    with open(bad_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
        
    engine = ShopRankServingEngine(model_dir=bad_dir, redis_url="disabled")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        engine.load_bundle()
