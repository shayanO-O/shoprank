import json
from pathlib import Path
import numpy as np
import pytest

from src.monitoring.drift import (
    DriftDetector,
    compute_categorical_chi2,
    compute_ks_statistic,
    compute_psi,
)

def test_compute_psi_stable_vs_shifted():
    rng = np.random.default_rng(42)
    # Identical distributions
    ref = rng.normal(loc=10.0, scale=2.0, size=2000)
    curr_stable = rng.normal(loc=10.0, scale=2.0, size=2000)
    psi_stable = compute_psi(ref, curr_stable, num_bins=10)
    assert psi_stable < 0.10

    # Heavily shifted distribution
    curr_shifted = rng.normal(loc=15.0, scale=3.0, size=2000)
    psi_shifted = compute_psi(ref, curr_shifted, num_bins=10)
    assert psi_shifted > 0.25

def test_negative_control_no_false_alarm():
    """
    Negative control: Two independent subsets from the same data source must NOT alarm.
    """
    rng = np.random.default_rng(100)
    base_popularity = rng.exponential(scale=50.0, size=1000)
    ref_counts = base_popularity[:500]
    curr_counts = base_popularity[500:]

    ref_scores = rng.normal(loc=0.5, scale=0.1, size=500)
    curr_scores = rng.normal(loc=0.5, scale=0.1, size=500)

    detector = DriftDetector(psi_threshold=0.25, ks_threshold=0.08)
    res = detector.evaluate_drift(
        ref_item_counts=ref_counts,
        curr_item_counts=curr_counts,
        ref_scores=ref_scores,
        curr_scores=curr_scores
    )

    assert res["is_drift_detected"] is False
    assert res["popularity_status"] in ("stable", "warning")
    assert res["score_status"] == "stable"
    assert len(res["reasons"]) == 0

def test_positive_control_drift_detected():
    """
    Positive control: Injected popularity skew MUST trigger drift alarm.
    """
    rng = np.random.default_rng(200)
    ref_counts = rng.exponential(scale=20.0, size=800)
    # Skew current counts by boosting tail (simulating heavy viral surge)
    curr_counts = ref_counts.copy()
    curr_counts[:300] *= 15.0  # massive viral surge

    detector = DriftDetector(psi_threshold=0.25)
    res = detector.evaluate_drift(ref_item_counts=ref_counts, curr_item_counts=curr_counts)

    assert res["is_drift_detected"] is True
    assert res["popularity_status"] == "alarm"
    assert any("PSI" in r for r in res["reasons"])

def test_score_distribution_ks_drift():
    rng = np.random.default_rng(300)
    ref_scores = rng.uniform(0.1, 0.9, size=500)
    # Significant score shift (e.g. uncalibrated scores)
    curr_scores = rng.uniform(0.4, 1.2, size=500)

    detector = DriftDetector(ks_threshold=0.08)
    res = detector.evaluate_drift(
        ref_item_counts=np.array([10]*100),
        curr_item_counts=np.array([10]*100),
        ref_scores=ref_scores,
        curr_scores=curr_scores
    )

    assert res["is_drift_detected"] is True
    assert res["score_status"] == "alarm"

def test_retraining_cooldown(tmp_path):
    state_file = tmp_path / "drift_state.json"
    detector = DriftDetector(cooldown_seconds=3600.0, state_file=state_file)

    drift_report = {
        "is_drift_detected": True,
        "reasons": ["Popularity PSI 0.35 >= 0.25"]
    }

    # First trigger -> succeeds
    res1 = detector.trigger_retraining(drift_report)
    assert res1["triggered"] is True
    assert res1["cooldown_active"] is False

    # Second trigger immediately -> suppressed by cooldown
    res2 = detector.trigger_retraining(drift_report)
    assert res2["triggered"] is False
    assert res2["cooldown_active"] is True
    assert res2["remaining_seconds"] > 0

def test_grafana_dashboard_json_validity():
    dash_path = Path("deploy/grafana/dashboard.json")
    assert dash_path.exists()
    
    with open(dash_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "title" in data
    assert "panels" in data
    assert len(data["panels"]) >= 5
    assert data["templating"]["list"][0]["name"] == "model_version"

    # Verify key Prometheus metric queries exist in panels
    json_str = json.dumps(data)
    assert "shoprank_recommendation_requests_total" in json_str
    assert "shoprank_recommendation_latency_seconds_bucket" in json_str
    assert "shoprank_shadow_errors_total" in json_str
