import json
from pathlib import Path
import numpy as np
import pytest

from src.pipeline.quality_gate import (
    QualityGate,
    compute_bootstrap_ndcg_ci,
    compute_rank_overlap,
)
from dags.shoprank_dag import dag as training_dag
from dags.shoprank_drift_retrain_dag import dag as drift_dag

def test_bootstrap_ndcg_ci():
    c_scores = np.array([0.15, 0.20, 0.18, 0.25, 0.12] * 20)
    champ_scores = np.array([0.10, 0.15, 0.12, 0.20, 0.08] * 20)
    
    mean_diff, ci_low, ci_high = compute_bootstrap_ndcg_ci(c_scores, champ_scores, n_bootstraps=500, seed=42)
    assert pytest.approx(mean_diff, rel=1e-2) == 0.05
    assert ci_low > 0.0
    assert ci_high > ci_low

    # Ties: identical scores
    m_diff, ci_l, ci_u = compute_bootstrap_ndcg_ci(c_scores, c_scores, n_bootstraps=100, seed=42)
    assert m_diff == 0.0
    assert ci_l == 0.0
    assert ci_u == 0.0

def test_quality_gate_approved():
    gate = QualityGate(min_ndcg_margin=0.005, max_p95_latency_ms=30.0, min_catalog_coverage=0.10)
    challenger_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.095, "catalog_coverage": 0.14},
            "latency": {"p95": 14.5}
        }
    }
    champ_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.082, "catalog_coverage": 0.12}
        }
    }
    shadow_metrics = {
        "sample_size": 1500,
        "error_rate": 0.0005  # 0.05% < 0.1%
    }
    
    res = gate.evaluate(
        challenger_results=challenger_res,
        champion_results=champ_res,
        shadow_metrics=shadow_metrics,
        sample_recs_challenger=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        sample_recs_champion=[1, 2, 3, 4, 5, 11, 12, 13, 14, 15]
    )
    
    assert res["qualified_for_promotion"] is True
    assert res["status"] == "APPROVED"
    assert len(res["failures"]) == 0

def test_quality_gate_tie_rejection():
    """
    Ties must be rejected (margin > 0.005 and CI lower bound > 0).
    """
    gate = QualityGate(min_ndcg_margin=0.005)
    res_tie = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.085, "catalog_coverage": 0.14},
            "latency": {"p95": 15.0}
        }
    }
    
    res = gate.evaluate(challenger_results=res_tie, champion_results=res_tie)
    assert res["qualified_for_promotion"] is False
    assert res["status"] == "REJECTED"
    assert any("NDCG diff" in f for f in res["failures"])

def test_quality_gate_latency_failure():
    gate = QualityGate(max_p95_latency_ms=30.0)
    c_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.095, "catalog_coverage": 0.14},
            "latency": {"p95": 38.5}  # exceeds 30ms
        }
    }
    champ_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.080, "catalog_coverage": 0.12}
        }
    }
    res = gate.evaluate(challenger_results=c_res, champion_results=champ_res)
    assert res["qualified_for_promotion"] is False
    assert any("latency" in f for f in res["failures"])

def test_quality_gate_shadow_insufficient_sample_size():
    gate = QualityGate(min_shadow_sample_size=1000)
    c_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.095, "catalog_coverage": 0.14},
            "latency": {"p95": 15.0}
        }
    }
    champ_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.080, "catalog_coverage": 0.12}
        }
    }
    # Only 200 shadow requests
    shadow = {"sample_size": 200, "error_rate": 0.0}
    res = gate.evaluate(challenger_results=c_res, champion_results=champ_res, shadow_metrics=shadow)
    assert res["qualified_for_promotion"] is False
    assert any("sample size" in f for f in res["failures"])

def test_quality_gate_shadow_high_error_rate():
    gate = QualityGate(max_shadow_error_rate=0.001)
    c_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.095, "catalog_coverage": 0.14},
            "latency": {"p95": 15.0}
        }
    }
    champ_res = {
        "retrieval_plus_ranker": {
            "test": {"ndcg_at_10": 0.080, "catalog_coverage": 0.12}
        }
    }
    # 2,000 requests but 0.5% error rate (> 0.1%)
    shadow = {"sample_size": 2000, "error_rate": 0.005}
    res = gate.evaluate(challenger_results=c_res, champion_results=champ_res, shadow_metrics=shadow)
    assert res["qualified_for_promotion"] is False
    assert any("error rate" in f for f in res["failures"])

def test_dag_structure_and_acyclic():
    # 1. Training pipeline DAG tasks
    tasks = training_dag.tasks
    assert "validate_raw_data" in tasks
    assert "preprocess_interactions" in tasks
    assert "split_dataset" in tasks
    assert "train_two_tower_retrieval" in tasks
    assert "build_faiss_index" in tasks
    assert "train_lambdarank_ranker" in tasks
    assert "evaluate_system" in tasks
    assert "register_model_bundle" in tasks
    assert "evaluate_quality_gate" in tasks

    # Verify build_index is between retrieval and ranker
    retrieval_op = tasks["train_two_tower_retrieval"]
    index_op = tasks["build_faiss_index"]
    ranker_op = tasks["train_lambdarank_ranker"]
    assert index_op in retrieval_op.downstream
    assert ranker_op in index_op.downstream

    # Verify register and quality_gate are explicit downstream of evaluate
    eval_op = tasks["evaluate_system"]
    reg_op = tasks["register_model_bundle"]
    qg_op = tasks["evaluate_quality_gate"]
    assert reg_op in eval_op.downstream
    assert qg_op in reg_op.downstream

    # 2. Drift Retraining DAG
    d_tasks = drift_dag.tasks
    assert "sync_production_interactions" in d_tasks
    assert "rebuild_index" in d_tasks
    assert "evaluate_gate" in d_tasks
