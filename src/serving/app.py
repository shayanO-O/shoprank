import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
import faiss
import lightgbm as lgb
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

load_dotenv()

from src.serving.business_rules import apply_business_rules
from src.serving.feature_store import OnlineFeatureStore
from src.serving.metrics_exporter import (
    CONTENT_TYPE_LATEST,
    get_metrics_content,
    record_recommendation_metrics,
)
from src.two_tower import UserTower
from src.register import calculate_file_sha256

logger = logging.getLogger(__name__)

# Request / Response Schemas

class RecommendRequest(BaseModel):
    user_id: int = Field(..., description="Target user ID")
    top_k: int = Field(10, ge=1, le=100, description="Number of recommendations to return")
    as_of: Optional[float] = Field(None, description="Point-in-time timestamp for simulation replays")

class RecommendationItem(BaseModel):
    item_id: int
    score: float
    rank: int

class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    model_version: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_version: str

class VersionResponse(BaseModel):
    model_version: str
    bundle_hash: str
    git_commit: str
    data_version: str
    artifacts: Dict[str, str]

# Serving Engine State Container

class ShopRankServingEngine:
    def __init__(self, model_dir: Path, redis_url: Optional[str] = None, model_version: str = "v0.1.0"):
        self.model_dir = Path(model_dir)
        self.redis_url = redis_url
        self.model_version = model_version
        
        self.user_tower: Optional[UserTower] = None
        self.index: Optional[faiss.Index] = None
        self.ranker: Optional[lgb.Booster] = None
        self.all_items: List[int] = []
        self.item_to_idx: Dict[int, int] = {}
        self.popular_items: List[int] = []
        self.manifest: Dict[str, Any] = {}
        self.is_ready: bool = False
        self.feature_store: Optional[OnlineFeatureStore] = None
        self.item_genres: Dict[int, np.ndarray] = {}
        self.unavailable_items: Set[int] = set()

    def load_bundle(self):
        logger.info(f"Loading ShopRank model bundle from {self.model_dir}...")
        
        # 1. Verify Manifest and Artifact Checksums
        manifest_path = self.model_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            
            # Verify file integrity
            expected_artifacts = self.manifest.get("artifacts", {})
            for art_name, expected_hash in expected_artifacts.items():
                art_file = self.model_dir / art_name
                if not art_file.exists():
                    raise FileNotFoundError(f"Missing artifact: {art_file}")
                curr_hash = calculate_file_sha256(art_file)
                if curr_hash != expected_hash:
                    raise ValueError(f"Checksum mismatch for {art_name}: expected {expected_hash}, got {curr_hash}")
            logger.info("Manifest checksums verified successfully.")
        else:
            logger.warning(f"No manifest.json found at {manifest_path}. Proceeding with unverified load.")
            self.manifest = {
                "bundle_hash": "unverified",
                "git_commit": "unknown",
                "data_version": "unknown",
                "artifacts": {}
            }

        # 2. Load Item ID Map
        map_path = self.model_dir / "item_id_map.json"
        with open(map_path, "r", encoding="utf-8") as f:
            self.all_items = [int(x) for x in json.load(f)]
        self.item_to_idx = {mid: i for i, mid in enumerate(self.all_items)}

        # 3. Load FAISS Index
        index_path = self.model_dir / "faiss.index"
        self.index = faiss.read_index(str(index_path))
        dim = self.index.d

        # 4. Load PyTorch User Tower
        self.user_tower = UserTower(num_items=len(self.all_items), emb_dim=dim, max_len=50)
        tt_path = self.model_dir / "two_tower.pt"
        st = torch.load(tt_path, weights_only=False, map_location="cpu")
        u_st = {k.replace("user_tower.", ""): v for k, v in st.items() if k.startswith("user_tower.")}
        self.user_tower.load_state_dict(u_st)
        self.user_tower.eval()

        # 5. Load LightGBM Ranker
        ranker_path = self.model_dir / "ranker.txt"
        self.ranker = lgb.Booster(model_file=str(ranker_path))

        # 6. Initialize Feature Store
        self.feature_store = OnlineFeatureStore(redis_url=self.redis_url)
        
        # Load availability and genres if available
        avail_path = Path("data/processed/availability.parquet")
        if avail_path.exists():
            try:
                import pandas as pd
                df_avail = pd.read_parquet(avail_path)
                unavail = df_avail[df_avail["is_available"] == False]["movieId"]
                self.unavailable_items = set(unavail.astype(int).tolist())
            except Exception as e:
                logger.warning(f"Could not load availability: {e}")

        # Popular fallback items (top 50)
        self.popular_items = self.all_items[:50]
        self.is_ready = True
        logger.info(f"ShopRank Serving Engine initialized. Model version: {self.model_version}")

    def retrieve_candidates(self, user_id: int, as_of: Optional[float] = None, top_k: int = 200) -> Tuple[List[int], Dict[int, float]]:
        """
        Stage 1: User Tower embedding + FAISS vector search (or popularity for cold users).
        """
        user_feats = self.feature_store.get_user_features(user_id, as_of=as_of, max_len=50)
        hist = user_feats["history"]
        seen = user_feats["seen_items"]

        if not hist:
            # Cold user fallback
            cands = [m for m in self.popular_items if m not in seen][:top_k]
            scores = {m: 0.0 for m in cands}
            return cands, scores

        # Build user embedding
        hist_idx = [self.item_to_idx[m] for m in hist if m in self.item_to_idx][-50:]
        hist_len = min(len(hist_idx), 50)
        padded = np.zeros((1, 50), dtype=np.int64)
        padded[0, :hist_len] = hist_idx
        
        with torch.no_grad():
            t_hist = torch.tensor(padded, dtype=torch.long)
            t_len = torch.tensor([hist_len], dtype=torch.float32)
            u_vec = self.user_tower(t_hist, t_len).cpu().numpy()

        # FAISS search
        k_search = min(top_k + len(seen) + 50, len(self.all_items))
        dists, indices = self.index.search(np.ascontiguousarray(u_vec, dtype=np.float32), k_search)
        
        cands = []
        scores = {}
        for score, idx in zip(dists[0], indices[0]):
            if idx < 0 or idx >= len(self.all_items):
                continue
            mid = self.all_items[idx]
            if mid not in seen:
                cands.append(mid)
                scores[mid] = float(score)
                if len(cands) >= top_k:
                    break

        return cands, scores

    def rank_candidates(self, user_id: int, candidates: List[int], cand_scores: Dict[int, float], as_of: Optional[float] = None) -> List[Tuple[int, float]]:
        """
        Stage 2: LightGBM LambdaRank re-ranking with point-in-time features.
        """
        if not candidates:
            return []

        user_feats = self.feature_store.get_user_features(user_id, as_of=as_of)
        user_genre_vec = user_feats["user_genre_vector"]
        u_norm = np.linalg.norm(user_genre_vec)
        hist_len = float(user_feats["history_len"])

        item_feats = self.feature_store.get_item_features(candidates)
        item_genres = self.feature_store.get_item_genres()
        num_genres = len(user_genre_vec) if len(user_genre_vec) > 0 else 20

        rows = []
        for mid in candidates:
            s = cand_scores.get(mid, 0.0)
            st = item_feats.get(mid, {})
            pop = st.get("popularity", 0.0)
            r_pop = st.get("recent_popularity", 0.0)
            age = st.get("item_age", 10.0)
            m_rat = st.get("mean_rating", 3.0)

            m_g = item_genres.get(mid, np.zeros(num_genres, dtype=np.float32))
            m_norm = np.linalg.norm(m_g)
            overlap = float(np.dot(user_genre_vec, m_g) / (u_norm * m_norm)) if (u_norm > 0 and m_norm > 0) else 0.0

            rows.append([s, pop, r_pop, age, m_rat, hist_len, overlap])

        x_val = np.array(rows, dtype=np.float32)
        pred_scores = self.ranker.predict(x_val)
        
        ranked_pairs = sorted(zip(candidates, pred_scores), key=lambda x: x[1], reverse=True)
        return ranked_pairs

    def recommend(self, user_id: int, top_k: int = 10, as_of: Optional[float] = None) -> Tuple[List[RecommendationItem], int]:
        """
        Full two-stage inference + business rules pipeline.
        """
        # 1. Retrieval
        candidates, cand_scores = self.retrieve_candidates(user_id, as_of=as_of, top_k=200)
        n_candidates = len(candidates)

        # 2. Ranking
        ranked_pairs = self.rank_candidates(user_id, candidates, cand_scores, as_of=as_of)
        ranked_ids = [mid for mid, _ in ranked_pairs]
        score_dict = {mid: score for mid, score in ranked_pairs}

        # 3. Business Rules Post-Processing
        seen = self.feature_store.get_user_seen(user_id, as_of=as_of)
        item_genres = self.feature_store.get_item_genres()
        final_ids = apply_business_rules(
            candidates=ranked_ids,
            seen_set=seen,
            unavailable_set=self.unavailable_items,
            item_genre_dict=item_genres,
            max_per_genre=3,
            top_k=top_k
        )

        recommendations = [
            RecommendationItem(item_id=mid, score=float(score_dict.get(mid, 0.0)), rank=idx + 1)
            for idx, mid in enumerate(final_ids)
        ]
        return recommendations, n_candidates

# Application Factory

def create_app(engine: Optional[ShopRankServingEngine] = None) -> FastAPI:
    app_engine = engine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal app_engine
        if app_engine is None:
            model_dir = Path(os.environ.get("MODEL_DIR", "models"))
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            model_ver = os.environ.get("MODEL_VERSION", "v0.1.0")
            app_engine = ShopRankServingEngine(model_dir=model_dir, redis_url=redis_url, model_version=model_ver)
            try:
                app_engine.load_bundle()
            except Exception as e:
                logger.error(f"Failed to load bundle on startup: {e}")
        yield

    app = FastAPI(title="ShopRank Serving API", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz():
        ver = app_engine.model_version if app_engine else "uninitialized"
        return HealthResponse(status="alive", model_version=ver)

    @app.get("/ready", response_model=HealthResponse)
    def ready():
        if app_engine is None or not app_engine.is_ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model bundle not ready")
        return HealthResponse(status="ready", model_version=app_engine.model_version)

    @app.get("/version", response_model=VersionResponse)
    def version():
        if app_engine is None or not app_engine.is_ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model bundle not ready")
        m = app_engine.manifest
        return VersionResponse(
            model_version=app_engine.model_version,
            bundle_hash=m.get("bundle_hash", "unknown"),
            git_commit=m.get("git_commit", "unknown"),
            data_version=m.get("data_version", "unknown"),
            artifacts=m.get("artifacts", {})
        )

    @app.post("/recommend", response_model=RecommendResponse)
    def recommend(req: RecommendRequest):
        if app_engine is None or not app_engine.is_ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model engine not ready")
        
        t0 = time.perf_counter()
        status_label = "success"
        try:
            recs, n_cands = app_engine.recommend(user_id=req.user_id, top_k=req.top_k, as_of=req.as_of)
        except Exception as e:
            status_label = "error"
            t1 = time.perf_counter()
            record_recommendation_metrics(
                model_version=app_engine.model_version,
                status=status_label,
                latency_seconds=t1 - t0,
                candidates_count=0,
                final_count=0
            )
            logger.error(f"Inference error for user {req.user_id}: {e}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

        t1 = time.perf_counter()
        latency_s = t1 - t0
        latency_ms = latency_s * 1000.0

        record_recommendation_metrics(
            model_version=app_engine.model_version,
            status=status_label,
            latency_seconds=latency_s,
            candidates_count=n_cands,
            final_count=len(recs)
        )

        return RecommendResponse(
            user_id=req.user_id,
            recommendations=recs,
            model_version=app_engine.model_version,
            latency_ms=latency_ms
        )

    @app.get("/metrics")
    def metrics():
        return Response(content=get_metrics_content(), media_type=CONTENT_TYPE_LATEST)

    return app

# Default top-level instance for ASGI servers
app = create_app()
