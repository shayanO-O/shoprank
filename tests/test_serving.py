import numpy as np
import pytest

from src.serving.feature_store import OnlineFeatureStore
from src.serving.business_rules import (
    filter_seen_items,
    filter_unavailable_items,
    enforce_diversity,
    trim_top_k,
    apply_business_rules,
)

def test_feature_store_point_in_time():
    # Use in-memory store by setting redis_url='disabled'
    fs = OnlineFeatureStore(redis_url="disabled")
    uid = 42
    
    # Record interactions with explicit timestamps
    fs.record_interaction(user_id=uid, item_id=101, timestamp=1000.0)
    fs.record_interaction(user_id=uid, item_id=102, timestamp=2000.0)
    fs.record_interaction(user_id=uid, item_id=103, timestamp=3000.0)
    fs.record_interaction(user_id=uid, item_id=104, timestamp=4000.0)
    
    # Query as of t=2500 -> should only see 101 and 102
    hist_2500 = fs.get_user_history(user_id=uid, as_of=2500.0)
    seen_2500 = fs.get_user_seen(user_id=uid, as_of=2500.0)
    assert hist_2500 == [101, 102]
    assert seen_2500 == {101, 102}
    
    # Query as of t=4500 -> should see all 4
    hist_all = fs.get_user_history(user_id=uid, as_of=4500.0)
    assert hist_all == [101, 102, 103, 104]
    
    # Query without as_of -> should see all 4
    hist_now = fs.get_user_history(user_id=uid)
    assert hist_now == [101, 102, 103, 104]

def test_feature_store_materialization():
    fs = OnlineFeatureStore(redis_url="disabled")
    uid = 99
    item = 555
    
    # Initial state
    features_before = fs.get_item_features([item])
    assert features_before[item]["popularity"] == 0.0
    
    # Record interaction
    fs.record_interaction(user_id=uid, item_id=item, timestamp=100.0)
    
    # Verify online state updated
    hist = fs.get_user_history(user_id=uid)
    assert hist == [item]
    
    features_after = fs.get_item_features([item])
    assert features_after[item]["popularity"] == 1.0
    
    # Record another interaction for same item
    fs.record_interaction(user_id=uid + 1, item_id=item, timestamp=200.0)
    features_after2 = fs.get_item_features([item])
    assert features_after2[item]["popularity"] == 2.0

def test_feature_store_cold_user_fallback():
    fs = OnlineFeatureStore(redis_url="disabled")
    cold_uid = 88888
    
    features = fs.get_user_features(cold_uid, num_genres=10)
    assert features["history"] == []
    assert features["seen_items"] == set()
    assert features["history_len"] == 0
    np.testing.assert_array_equal(features["user_genre_vector"], np.zeros(10, dtype=np.float32))

def test_filter_seen_items():
    cands = [10, 20, 30, 40, 50]
    seen = {20, 40}
    filtered = filter_seen_items(cands, seen)
    assert filtered == [10, 30, 50]

def test_filter_unavailable_items():
    cands = [10, 20, 30, 40, 50]
    unavail = {10, 50}
    filtered = filter_unavailable_items(cands, unavail)
    assert filtered == [20, 30, 40]

def test_enforce_diversity():
    # Genre 0 has 4 items, Genre 1 has 3 items
    genre_dict = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([1.0, 0.0]),
        4: np.array([1.0, 0.0]),
        5: np.array([0.0, 1.0]),
        6: np.array([0.0, 1.0]),
        7: np.array([0.0, 1.0]),
    }
    ranked = [1, 2, 3, 4, 5, 6, 7]
    # Cap at 2 per genre, top_k = 4
    diversified = enforce_diversity(ranked, genre_dict, max_per_genre=2, top_k=4)
    # Genre 0 gives 1, 2. Then Genre 1 gives 5, 6
    assert diversified == [1, 2, 5, 6]

def test_apply_business_rules_pipeline():
    cands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    seen = {1, 3}  # remaining: [2, 4, 5, 6, 7, 8, 9, 10]
    unavail = {2}  # remaining: [4, 5, 6, 7, 8, 9, 10]
    
    # Let 4, 5, 6, 7 belong to genre 0; 8, 9, 10 belong to genre 1
    genre_dict = {
        4: np.array([1.0, 0.0]),
        5: np.array([1.0, 0.0]),
        6: np.array([1.0, 0.0]),
        7: np.array([1.0, 0.0]),
        8: np.array([0.0, 1.0]),
        9: np.array([0.0, 1.0]),
        10: np.array([0.0, 1.0]),
    }
    
    # Cap genre 0 at 2 items, top_k=4
    final = apply_business_rules(
        candidates=cands,
        seen_set=seen,
        unavailable_set=unavail,
        item_genre_dict=genre_dict,
        max_per_genre=2,
        top_k=4
    )
    # Filtered order: 4, 5 (genre 0 capped), 8, 9 (genre 1)
    assert final == [4, 5, 8, 9]
