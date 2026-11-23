import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.serving.feature_store import OnlineFeatureStore
from src.serving.simulator import ClickModel, ReplaySimulator

def test_click_model_examination_decay():
    cm = ClickModel()
    p1 = cm.examination_probability(1)
    # 0.70 / log2(2) = 0.70
    assert pytest.approx(p1, rel=1e-3) == 0.70

    probs = [cm.examination_probability(k) for k in range(1, 11)]
    # Must be strictly decreasing
    for i in range(len(probs) - 1):
        assert probs[i] > probs[i + 1]

def test_click_model_relevance_calibration():
    cm = ClickModel(base_unrated_p=0.05)
    
    # Rating 3.5 -> sigmoid(0) = 0.50
    assert pytest.approx(cm.relevance_probability(3.5), rel=1e-3) == 0.50
    # Rating 5.0 -> sigmoid(1.5) approx 0.8175
    assert cm.relevance_probability(5.0) > 0.80
    # Rating 1.0 -> sigmoid(-2.5) approx 0.0758
    assert cm.relevance_probability(1.0) < 0.10
    # Unrated item policy -> fixed 0.05
    assert cm.relevance_probability(None) == 0.05
    assert cm.relevance_probability(math.nan) == 0.05

def test_click_model_drift_injection():
    # 1. Genre shift test
    cm_genre = ClickModel(
        drift_mode="genre_shift",
        drift_config={"target_genre_idx": 1, "boost_multiplier": 3.0}
    )
    g_target = np.array([0.0, 1.0, 0.0])
    g_other = np.array([1.0, 0.0, 0.0])
    assert cm_genre.get_drift_multiplier(item_id=10, item_genres=g_target) == 3.0
    assert cm_genre.get_drift_multiplier(item_id=11, item_genres=g_other) == 1.0

    # 2. Viral item test
    cm_viral = ClickModel(
        drift_mode="viral_item",
        drift_config={"viral_item_ids": [999], "boost_multiplier": 5.0}
    )
    assert cm_viral.get_drift_multiplier(item_id=999) == 5.0
    assert cm_viral.get_drift_multiplier(item_id=100) == 1.0

def test_replay_simulator_in_process(tmp_path):
    # Synthetic interactions
    data = {
        "userId": [1, 1, 2, 2, 3, 3],
        "movieId": [101, 102, 103, 104, 105, 106],
        "rating": [5.0, 4.0, 3.5, 2.0, 5.0, 4.5],
        "timestamp": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0]
    }
    df = pd.DataFrame(data)
    
    fs = OnlineFeatureStore(redis_url="disabled")
    
    # Mock recommend callable
    def mock_recommend(uid, top_k, as_of):
        return {
            "user_id": uid,
            "recommendations": [
                {"item_id": 101, "score": 0.9, "rank": 1},
                {"item_id": 102, "score": 0.8, "rank": 2},
                {"item_id": 999, "score": 0.7, "rank": 3}
            ],
            "model_version": "test-v1"
        }

    sim = ReplaySimulator(
        interactions_df=df,
        recommend_fn=mock_recommend,
        feature_store=fs,
        seed=42
    )

    out_file = tmp_path / "sim_out.json"
    res = sim.run_simulation(n_requests=4, top_k=3, output_log_path=str(out_file))

    assert res["total_requests"] == 4
    assert res["total_impressions"] == 12
    assert out_file.exists()

    # Read output log
    with open(out_file, "r") as f:
        log_data = json.load(f)
    assert len(log_data["events"]) == 4
    assert "clicked" in log_data["events"][0]
