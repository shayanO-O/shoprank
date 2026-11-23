import json
import time
import pytest
from fastapi.testclient import TestClient
import httpx

from src.serving.router import (
    RouterConfig,
    TrafficRouter,
    create_router_app,
    is_canary_user,
)

def test_sticky_hashing_consistency():
    uid = 42
    salt = "test-salt"
    # Same user always gets same assignment for a given weight
    assigned_1 = is_canary_user(uid, weight=30, salt=salt)
    assigned_2 = is_canary_user(uid, weight=30, salt=salt)
    assert assigned_1 == assigned_2

    # Weight boundary checks
    assert not is_canary_user(uid, weight=0, salt=salt)
    assert is_canary_user(uid, weight=100, salt=salt)

    # Statistical distribution check over 1000 users at weight=25
    canary_count = sum(1 for u in range(1000) if is_canary_user(u, weight=25, salt=salt))
    # 25% ± 5%
    assert 200 <= canary_count <= 300

def test_router_status_and_config():
    config = RouterConfig(canary_weight=15, shadow_enabled=True)
    router = TrafficRouter(config=config)
    status = router.get_status()
    
    assert status.canary_weight == 15
    assert status.shadow_enabled is True
    assert status.rollback_triggered is False

def test_rollback_watcher_error_trigger():
    config = RouterConfig(
        canary_weight=20,
        max_error_rate=0.05,  # 5% threshold
        min_sample_size=10
    )
    router = TrafficRouter(config=config)
    
    # Send 8 success calls
    for _ in range(8):
        router.record_canary_call(is_error=False, latency_ms=10.0)
    assert router.rollback_triggered is False
    assert router.config.canary_weight == 20
    
    # Send 3 error calls (total 11 calls, 3 errors = 27.2% > 5%)
    for _ in range(3):
        router.record_canary_call(is_error=True, latency_ms=10.0)
        
    assert router.rollback_triggered is True
    assert router.config.canary_weight == 0
    assert "exceeded threshold" in router.rollback_reason

def test_rollback_watcher_latency_trigger():
    config = RouterConfig(
        canary_weight=20,
        max_p95_latency_ms=25.0,  # 25ms threshold
        min_sample_size=10
    )
    router = TrafficRouter(config=config)
    
    # Send 10 calls with high latency (50ms)
    for _ in range(10):
        router.record_canary_call(is_error=False, latency_ms=50.0)
        
    assert router.rollback_triggered is True
    assert router.config.canary_weight == 0
    assert "latency" in router.rollback_reason

def test_router_endpoints_mocked():
    # Use mock transport for httpx.AsyncClient to simulate champion & challenger
    def mock_handler(request: httpx.Request):
        url = str(request.url)
        if "8000" in url:
            # Champion response
            return httpx.Response(200, json={
                "user_id": 1,
                "recommendations": [{"item_id": 101, "score": 0.9, "rank": 1}],
                "model_version": "champion-v1",
                "latency_ms": 5.0
            })
        elif "8001" in url:
            # Challenger response
            return httpx.Response(200, json={
                "user_id": 1,
                "recommendations": [{"item_id": 202, "score": 0.95, "rank": 1}],
                "model_version": "challenger-v2",
                "latency_ms": 6.0
            })
        return httpx.Response(404)

    mock_transport = httpx.MockTransport(mock_handler)
    router = TrafficRouter(config=RouterConfig(canary_weight=0, shadow_enabled=True))
    router.http_client = httpx.AsyncClient(transport=mock_transport)
    
    app = create_router_app(router=router)
    client = TestClient(app)
    
    # Health check
    res_h = client.get("/healthz")
    assert res_h.status_code == 200
    
    # Route request (weight=0 -> champion)
    res_rec = client.post("/recommend", json={"user_id": 1234, "top_k": 1})
    assert res_rec.status_code == 200
    assert res_rec.json()["model_version"] == "champion-v1"
    assert router.total_champion_requests == 1
    # Check shadow queue received task
    assert router.shadow_queue.qsize() == 1
    
    # Update config to 100% canary
    res_cfg = client.post("/routing/config", json={
        "champion_url": "http://localhost:8000",
        "challenger_url": "http://localhost:8001",
        "canary_weight": 100,
        "shadow_enabled": False,
        "canary_salt": "test-salt",
        "max_error_rate": 0.05,
        "max_p95_latency_ms": 30.0,
        "min_sample_size": 10
    })
    assert res_cfg.status_code == 200
    assert router.config.canary_weight == 100
    
    # Route request (weight=100 -> challenger)
    res_can = client.post("/recommend", json={"user_id": 1234, "top_k": 1})
    assert res_can.status_code == 200
    assert res_can.json()["model_version"] == "challenger-v2"
    assert router.total_canary_requests == 1
