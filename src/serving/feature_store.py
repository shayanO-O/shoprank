import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class OnlineFeatureStore:
    """
    Production Feature Store supporting Redis with automatic graceful fallback
    to an in-memory dictionary cache.
    
    Guarantees:
    - Point-in-time correctness (as_of timestamp filtering) to prevent data leakage during replays.
    - Active materialization write path (record_interaction) to update online state dynamically.
    - Zero-history default fallbacks for cold users and new items.
    """
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client = None
        self._is_redis_active = False
        
        # In-memory storage structures (used as cache or fallback)
        # user_id -> List[Tuple[item_id, timestamp]]
        self._user_interactions: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # item_id -> dict of features
        self._item_features: Dict[int, Dict[str, Any]] = {}
        # item_id -> np.ndarray genre vector
        self._item_genres: Dict[int, np.ndarray] = {}
        # set of unavailable item_ids
        self._unavailable_items: Set[int] = set()
        
        self._connect_redis()

    def _connect_redis(self):
        if self.redis_url and self.redis_url.lower() != "disabled":
            try:
                import redis
                client = redis.Redis.from_url(self.redis_url, socket_timeout=1.0, socket_connect_timeout=1.0)
                client.ping()
                self.redis_client = client
                self._is_redis_active = True
                logger.info(f"Connected to Redis feature store at {self.redis_url}")
            except Exception as e:
                self.redis_client = None
                self._is_redis_active = False
                logger.warning(f"Redis unavailable ({e}). Using in-memory fallback feature store.")
        else:
            self._is_redis_active = False
            logger.info("Redis disabled. Using in-memory feature store.")

    @property
    def is_redis_active(self) -> bool:
        return self._is_redis_active

    def record_interaction(self, user_id: int, item_id: int, timestamp: Optional[float] = None):
        """
        Active online write path: updates user history and rolling item popularity counters.
        """
        ts = float(timestamp if timestamp is not None else time.time())
        
        # Update in-memory user interactions
        self._user_interactions[user_id].append((item_id, ts))
        
        # Update item popularity
        if item_id in self._item_features:
            self._item_features[item_id]["popularity"] = self._item_features[item_id].get("popularity", 0.0) + 1.0
            self._item_features[item_id]["recent_popularity"] = self._item_features[item_id].get("recent_popularity", 0.0) + 1.0
        else:
            self._item_features[item_id] = {
                "popularity": 1.0,
                "recent_popularity": 1.0,
                "mean_rating": 3.0,
                "item_age": 0.0
            }

        # If Redis is active, sync to Redis
        if self._is_redis_active and self.redis_client:
            try:
                u_key = f"user:{user_id}:history"
                # Store as zset: value=item_id, score=timestamp
                self.redis_client.zadd(u_key, {str(item_id): ts})
                # Increment item popularity
                pop_key = f"item:{item_id}:stats"
                self.redis_client.hincrbyfloat(pop_key, "popularity", 1.0)
                self.redis_client.hincrbyfloat(pop_key, "recent_popularity", 1.0)
            except Exception as e:
                logger.warning(f"Redis write error: {e}")

    def get_user_history(self, user_id: int, as_of: Optional[float] = None, max_len: int = 50) -> List[int]:
        """
        Point-in-time user interaction history.
        Strictly guarantees: timestamp <= as_of (prevents future leakage).
        """
        if self._is_redis_active and self.redis_client:
            try:
                u_key = f"user:{user_id}:history"
                max_score = as_of if as_of is not None else "+inf"
                # Get items up to max_score in reverse order
                items = self.redis_client.zrevrangebyscore(u_key, max_score, "-inf", start=0, num=max_len)
                if items:
                    # Reverse to return chronological order
                    return [int(m) for m in reversed(items)]
            except Exception as e:
                logger.warning(f"Redis read error: {e}")

        # In-memory fallback
        raw_hist = self._user_interactions.get(user_id, [])
        if as_of is not None:
            filtered = [m for m, ts in raw_hist if ts <= as_of]
        else:
            filtered = [m for m, _ in raw_hist]
            
        return filtered[-max_len:]

    def get_user_seen(self, user_id: int, as_of: Optional[float] = None) -> Set[int]:
        """
        Set of all item IDs interacted with by the user up to as_of.
        """
        if self._is_redis_active and self.redis_client:
            try:
                u_key = f"user:{user_id}:history"
                max_score = as_of if as_of is not None else "+inf"
                items = self.redis_client.zrangebyscore(u_key, "-inf", max_score)
                if items:
                    return {int(m) for m in items}
            except Exception as e:
                logger.warning(f"Redis read error: {e}")

        raw_hist = self._user_interactions.get(user_id, [])
        if as_of is not None:
            return {m for m, ts in raw_hist if ts <= as_of}
        return {m for m, _ in raw_hist}

    def get_user_features(self, user_id: int, as_of: Optional[float] = None, max_len: int = 50, num_genres: int = 20) -> Dict[str, Any]:
        """
        Returns full feature bundle for user tower inference:
        - history: List[int]
        - seen_items: Set[int]
        - user_genre_vector: np.ndarray (mean of history item genre vectors)
        """
        hist = self.get_user_history(user_id, as_of=as_of, max_len=max_len)
        seen = self.get_user_seen(user_id, as_of=as_of)
        
        user_genre = np.zeros(num_genres, dtype=np.float32)
        if hist and self._item_genres:
            genre_list = [self._item_genres[m] for m in hist if m in self._item_genres]
            if genre_list:
                user_genre = np.mean(genre_list, axis=0).astype(np.float32)
                
        return {
            "history": hist,
            "seen_items": seen,
            "user_genre_vector": user_genre,
            "history_len": len(hist)
        }

    def get_item_features(self, item_ids: List[int], default_stats: Optional[Dict[str, float]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Retrieves ranking features for a batch of candidate items.
        Falls back to default_stats for cold/unseen items.
        """
        defaults = default_stats or {
            "popularity": 0.0,
            "recent_popularity": 0.0,
            "mean_rating": 3.0,
            "item_age": 10.0
        }
        
        result = {}
        for mid in item_ids:
            st = self._item_features.get(mid)
            if st is None:
                st = dict(defaults)
            result[mid] = st
        return result

    def get_item_genres(self) -> Dict[int, np.ndarray]:
        return self._item_genres

    def get_unavailable_items(self) -> Set[int]:
        return self._unavailable_items

    def seed_from_data(self,
                        train_df: Optional[pd.DataFrame] = None,
                        items_df: Optional[pd.DataFrame] = None,
                        availability_df: Optional[pd.DataFrame] = None,
                        genre_matrix: Optional[np.ndarray] = None):
        """
        Seeds the feature store with processed training data and item metadata.
        """
        if items_df is not None:
            for i, row in items_df.iterrows():
                mid = int(row["movieId"])
                self._item_features[mid] = {
                    "popularity": 0.0,
                    "recent_popularity": 0.0,
                    "mean_rating": float(row.get("mean_rating", 3.0)),
                    "item_age": float(row.get("item_age", 10.0))
                }
                if genre_matrix is not None and i < len(genre_matrix):
                    self._item_genres[mid] = np.array(genre_matrix[i], dtype=np.float32)

        if train_df is not None:
            # Seed histories
            for uid, mid, ts in zip(train_df["userId"], train_df["movieId"], train_df["timestamp"]):
                self._user_interactions[int(uid)].append((int(mid), float(ts)))
                
            # Compute popularity counts
            pop_counts = train_df["movieId"].value_counts()
            for mid, cnt in pop_counts.items():
                mid_int = int(mid)
                if mid_int in self._item_features:
                    self._item_features[mid_int]["popularity"] = float(cnt)
                    self._item_features[mid_int]["recent_popularity"] = float(cnt)

        if availability_df is not None:
            unavail = availability_df[availability_df["is_available"] == False]["movieId"]
            self._unavailable_items = set(unavail.astype(int).tolist())

    def verify_feature_parity(self, user_id: int, expected_hist: List[int], as_of: Optional[float] = None) -> bool:
        """
        Validates that online lookup produces identical history to offline pipeline.
        """
        actual_hist = self.get_user_history(user_id, as_of=as_of, max_len=len(expected_hist))
        return actual_hist == expected_hist
