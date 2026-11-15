import sys
from pathlib import Path
import numpy as np

# Support Root Module Discovery In Test Environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import ndcg_at_k, recall_at_k, catalog_coverage, intra_list_diversity, calculate_latency_percentiles

# Test Perfect And Empty NDCG Values
def test_ndcg_at_k():
    actual = [10, 20, 30]
    perfect_recs = [10, 20, 30, 40, 50]
    assert np.isclose(ndcg_at_k(actual, perfect_recs, k=10), 1.0)
    
    empty_actual = []
    assert ndcg_at_k(empty_actual, perfect_recs, k=10) == 0.0
    
    no_hit_recs = [100, 200, 300]
    assert ndcg_at_k(actual, no_hit_recs, k=10) == 0.0

# Test Recall Calculation Edge Cases
def test_recall_at_k():
    actual = [1, 2, 3, 4]
    recs = [1, 2, 5, 6]
    assert np.isclose(recall_at_k(actual, recs, k=4), 0.5)
    assert recall_at_k([], recs, k=4) == 0.0

# Test Catalog Coverage Computation
def test_catalog_coverage():
    all_recs = [[1, 2, 3], [3, 4, 5]]
    catalog_size = 10
    cov = catalog_coverage(all_recs, catalog_size)
    assert np.isclose(cov, 0.5)

# Test Intra List Genre Diversity
def test_intra_list_diversity():
    item_genres = {
        1: np.array([1.0, 0.0], dtype=np.float32),
        2: np.array([0.0, 1.0], dtype=np.float32)
    }
    all_recs = [[1, 2]]
    div = intra_list_diversity(all_recs, item_genres)
    assert np.isclose(div, 1.0)

# Test Latency Percentile Calculation
def test_latency_percentiles():
    times = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = calculate_latency_percentiles(times)
    assert np.isclose(stats["p50"], 30.0)
    assert stats["p95"] >= 40.0
