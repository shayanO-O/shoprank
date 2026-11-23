import argparse
import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import httpx
import numpy as np
import pandas as pd

from src.serving.feature_store import OnlineFeatureStore

logger = logging.getLogger(__name__)

# Calibrated Examination & Click Model

class ClickModel:
    """
    Examination x Relevance click decomposition model with support for
    unrated item baseline policy and controlled drift injection.
    
    Formula:
    P(examine | rank k) = 0.70 / log2(k + 1)
    P(rel | rating) = 1 / (1 + exp(-(rating - 3.5)))
    P(rel | unrated) = P0 (default 0.05)
    P(click) = P(examine | k) * P(rel) * drift_multiplier
    """
    def __init__(
        self,
        base_unrated_p: float = 0.05,
        drift_mode: str = "none",
        drift_config: Optional[Dict[str, Any]] = None,
        seed: int = 42
    ):
        self.base_unrated_p = base_unrated_p
        self.drift_mode = drift_mode
        self.drift_config = drift_config or {}
        self.rng = np.random.default_rng(seed)

    def examination_probability(self, rank: int) -> float:
        """
        Rank k is 1-indexed (1, 2, ..., 10).
        """
        if rank < 1:
            rank = 1
        return float(0.70 / math.log2(rank + 1))

    def relevance_probability(self, rating: Optional[float]) -> float:
        if rating is None or math.isnan(rating):
            return self.base_unrated_p
        # Sigmoid centered at 3.5
        return float(1.0 / (1.0 + math.exp(-(rating - 3.5))))

    def get_drift_multiplier(
        self,
        item_id: int,
        item_genres: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> float:
        start_ts = self.drift_config.get("start_timestamp", 0.0)
        if timestamp is not None and timestamp < start_ts:
            return 1.0

        if self.drift_mode == "genre_shift":
            target_genre_idx = self.drift_config.get("target_genre_idx", 0)
            boost = self.drift_config.get("boost_multiplier", 2.0)
            if item_genres is not None and len(item_genres) > target_genre_idx:
                if item_genres[target_genre_idx] > 0:
                    return boost
            return 1.0

        elif self.drift_mode == "viral_item":
            viral_item_ids = set(self.drift_config.get("viral_item_ids", []))
            boost = self.drift_config.get("boost_multiplier", 5.0)
            if item_id in viral_item_ids:
                return boost
            return 1.0

        return 1.0

    def simulate_clicks(
        self,
        recommendations: List[int],
        user_ratings_map: Dict[int, float],
        item_genre_dict: Optional[Dict[int, np.ndarray]] = None,
        timestamp: Optional[float] = None
    ) -> List[int]:
        """
        Returns list of clicked item IDs for the recommendations.
        """
        clicked = []
        for idx, item_id in enumerate(recommendations):
            rank = idx + 1
            p_exam = self.examination_probability(rank)
            rating = user_ratings_map.get(item_id)
            p_rel = self.relevance_probability(rating)
            
            g_vec = item_genre_dict.get(item_id) if item_genre_dict else None
            mult = self.get_drift_multiplier(item_id, item_genres=g_vec, timestamp=timestamp)
            
            p_click = min(1.0, p_exam * p_rel * mult)
            if self.rng.random() < p_click:
                clicked.append(item_id)

        return clicked

# Chronological Replay Simulator

class ReplaySimulator:
    """
    Replays request timelines chronologically against an active serving endpoint
    or in-process recommendation callable. Updates feature store upon clicks
    to simulate real online feedback loops.
    """
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        recommend_fn: Optional[Callable[[int, int, Optional[float]], Dict[str, Any]]] = None,
        endpoint_url: Optional[str] = None,
        feature_store: Optional[OnlineFeatureStore] = None,
        item_genre_dict: Optional[Dict[int, np.ndarray]] = None,
        click_model: Optional[ClickModel] = None,
        seed: int = 42
    ):
        self.df = interactions_df.sort_values("timestamp").reset_index(drop=True)
        self.recommend_fn = recommend_fn
        self.endpoint_url = endpoint_url
        self.feature_store = feature_store
        self.item_genre_dict = item_genre_dict or {}
        self.click_model = click_model or ClickModel(seed=seed)
        self.http_client = httpx.Client(timeout=1.0) if endpoint_url else None
        
        # User ground truth ratings map: user_id -> {movieId: rating}
        self.user_ratings: Dict[int, Dict[int, float]] = defaultdict(dict)
        for _, row in self.df.iterrows():
            uid = int(row["userId"])
            mid = int(row["movieId"])
            rat = float(row.get("rating", 4.0))
            self.user_ratings[uid][mid] = rat

    def close(self):
        if self.http_client:
            self.http_client.close()

    def _query_recommendations(self, user_id: int, top_k: int, as_of: float) -> Tuple[List[int], str, float]:
        t0 = time.perf_counter()
        if self.recommend_fn:
            res = self.recommend_fn(user_id, top_k, as_of)
            t1 = time.perf_counter()
            items = [item["item_id"] for item in res.get("recommendations", [])]
            ver = res.get("model_version", "unknown")
            lat_ms = (t1 - t0) * 1000.0
            return items, ver, lat_ms

        elif self.endpoint_url and self.http_client:
            resp = self.http_client.post(
                self.endpoint_url,
                json={"user_id": user_id, "top_k": top_k, "as_of": as_of}
            )
            t1 = time.perf_counter()
            if resp.status_code == 200:
                data = resp.json()
                items = [item["item_id"] for item in data.get("recommendations", [])]
                ver = data.get("model_version", "unknown")
                lat_ms = (t1 - t0) * 1000.0
                return items, ver, lat_ms
            else:
                logger.warning(f"Serving query failed: {resp.status_code}")
                return [], "error", (t1 - t0) * 1000.0
        else:
            raise ValueError("Neither recommend_fn nor endpoint_url provided")

    def run_simulation(
        self,
        n_requests: int = 1000,
        top_k: int = 10,
        output_log_path: Optional[str] = "metrics/simulation_log.json"
    ) -> Dict[str, Any]:
        """
        Executes chronological replay across n_requests unique (user_id, timestamp) events.
        """
        # Deduplicate requests by (userId, timestamp)
        requests = self.df.drop_duplicates(subset=["userId", "timestamp"]).head(n_requests)
        
        events = []
        total_impressions = 0
        total_clicks = 0
        all_recommended_items: Set[int] = set()
        clicked_item_counts: Dict[int, int] = defaultdict(int)

        logger.info(f"Starting chronological replay for {len(requests)} requests...")

        for idx, row in requests.iterrows():
            uid = int(row["userId"])
            ts = float(row["timestamp"])

            # 1. Query recommendations point-in-time
            recs, model_ver, lat_ms = self._query_recommendations(user_id=uid, top_k=top_k, as_of=ts)
            if not recs:
                continue

            total_impressions += len(recs)
            all_recommended_items.update(recs)

            # 2. Simulate user clicks
            ratings_map = self.user_ratings.get(uid, {})
            clicked = self.click_model.simulate_clicks(
                recommendations=recs,
                user_ratings_map=ratings_map,
                item_genre_dict=self.item_genre_dict,
                timestamp=ts
            )
            total_clicks += len(clicked)
            for mid in clicked:
                clicked_item_counts[mid] += 1

            # 3. Active Materialization: update feature store
            if self.feature_store and clicked:
                for mid in clicked:
                    self.feature_store.record_interaction(user_id=uid, item_id=mid, timestamp=ts)

            events.append({
                "timestamp": ts,
                "user_id": uid,
                "recommendations": recs,
                "clicked": clicked,
                "model_version": model_ver,
                "latency_ms": lat_ms
            })

        ctr = (total_clicks / total_impressions) if total_impressions > 0 else 0.0
        
        summary = {
            "total_requests": len(events),
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "overall_ctr": float(ctr),
            "unique_items_recommended": len(all_recommended_items),
            "unique_items_clicked": len(clicked_item_counts),
            "drift_mode": self.click_model.drift_mode,
            "top_clicked_items": sorted(clicked_item_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "events": events
        }

        if output_log_path:
            out_p = Path(output_log_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Replay log written to {out_p}")

        return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chronological Replay Simulator")
    parser.add_argument("--data-path", type=str, default="data/processed/test.parquet", help="Path to evaluation interactions")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/recommend", help="Serving endpoint")
    parser.add_argument("--n-requests", type=int, default=1000, help="Number of requests to replay")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K recommendations")
    parser.add_argument("--drift-mode", type=str, choices=["none", "genre_shift", "viral_item"], default="none")
    parser.add_argument("--output", type=str, default="metrics/simulation_log.json", help="Output path for simulation log")
    args = parser.parse_args()

    df_data = pd.read_parquet(args.data_path) if Path(args.data_path).exists() else pd.DataFrame()
    if df_data.empty:
        print(f"Data file not found at {args.data_path}. Please provide valid dataset.")
        exit(1)

    click_m = ClickModel(drift_mode=args.drift_mode)
    sim = ReplaySimulator(
        interactions_df=df_data,
        endpoint_url=args.endpoint,
        click_model=click_m
    )
    res = sim.run_simulation(n_requests=args.n_requests, top_k=args.top_k, output_log_path=args.output)
    print(f"Simulation completed: {res['total_requests']} requests, CTR: {res['overall_ctr'] * 100:.2f}%")
