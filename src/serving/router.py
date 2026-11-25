import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Support direct script execution and module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

load_dotenv()

from src.serving.app import RecommendRequest, RecommendResponse
from src.serving.metrics_exporter import (
    CONTENT_TYPE_LATEST,
    SHADOW_ERRORS,
    SHADOW_LATENCY,
    get_metrics_content,
)

logger = logging.getLogger(__name__)

# Router Configuration & State

class RouterConfig(BaseModel):
    champion_url: str = "http://localhost:8000"
    challenger_url: str = "http://localhost:8001"
    canary_weight: int = Field(0, ge=0, le=100, description="Canary traffic percentage [0-100]")
    shadow_enabled: bool = True
    canary_salt: str = "shoprank-canary-2026"
    max_error_rate: float = Field(0.01, description="Rollback threshold: max allowable error rate (1%)")
    max_p95_latency_ms: float = Field(30.0, description="Rollback threshold: max p95 latency (30ms)")
    min_sample_size: int = Field(50, description="Minimum requests before rollback trigger evaluates")

class RouterStatus(BaseModel):
    canary_weight: int
    shadow_enabled: bool
    rollback_triggered: bool
    rollback_reason: Optional[str]
    total_champion_requests: int
    total_canary_requests: int
    total_shadow_requests: int
    total_shadow_drops: int
    canary_recent_errors: int
    canary_recent_requests: int
    canary_recent_error_rate: float
    canary_recent_p95_latency_ms: float

# Sticky Hashing Helper

def is_canary_user(user_id: int, weight: int, salt: str = "shoprank-canary-2026") -> bool:
    """
    Deterministic sticky assignment:
    SHA-256(f"{user_id}:{salt}") mod 100 < weight.
    Guarantees user session stickiness without flipping models.
    """
    if weight <= 0:
        return False
    if weight >= 100:
        return True
    key = f"{user_id}:{salt}".encode("utf-8")
    hash_val = int(hashlib.sha256(key).hexdigest(), 16)
    return (hash_val % 100) < weight

# Traffic Router Engine

class TrafficRouter:
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig(
            champion_url=os.environ.get("CHAMPION_URL", "http://localhost:8000"),
            challenger_url=os.environ.get("CHALLENGER_URL", "http://localhost:8001"),
            canary_weight=int(os.environ.get("CANARY_WEIGHT", "0")),
            shadow_enabled=os.environ.get("SHADOW_ENABLED", "true").lower() in ("1", "true", "yes"),
            canary_salt=os.environ.get("CANARY_SALT", "shoprank-canary-2026"),
            max_error_rate=float(os.environ.get("MAX_ERROR_RATE", "0.01")),
            max_p95_latency_ms=float(os.environ.get("MAX_P95_LATENCY_MS", "30.0")),
            min_sample_size=int(os.environ.get("MIN_SAMPLE_SIZE", "50"))
        )
        
        # Internal counters
        self.total_champion_requests = 0
        self.total_canary_requests = 0
        self.total_shadow_requests = 0
        self.total_shadow_drops = 0
        self.rollback_triggered = False
        self.rollback_reason: Optional[str] = None
        
        # Sliding window for canary health: stores (timestamp, is_error, latency_ms)
        self._canary_window: deque = deque(maxlen=500)
        self._lock = threading.Lock()
        
        # Asynchronous shadow worker queue (bounded at 1000 items)
        self.shadow_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self._shadow_thread: Optional[threading.Thread] = None
        
        # HTTP client
        self.http_client: Optional[httpx.AsyncClient] = None

    def start_shadow_worker(self):
        self._stop_event.clear()
        SHADOW_ERRORS.labels(model_version="challenger-shadow")
        SHADOW_LATENCY.labels(model_version="challenger-shadow")
        self._shadow_thread = threading.Thread(target=self._shadow_worker_loop, daemon=True)
        self._shadow_thread.start()
        logger.info("Shadow inference worker thread started.")

    def stop_shadow_worker(self):
        self._stop_event.set()
        if self._shadow_thread and self._shadow_thread.is_alive():
            self._shadow_thread.join(timeout=2.0)
            logger.info("Shadow inference worker thread stopped.")

    def _shadow_worker_loop(self):
        """
        Background worker draining shadow requests.
        Runs synchronous HTTP requests in separate thread so champion latency is untouched.
        """
        sync_client = httpx.Client(timeout=0.08)
        while not self._stop_event.is_set():
            try:
                payload = self.shadow_queue.get(timeout=0.2)
            except queue.Empty:
                continue
                
            self.total_shadow_requests += 1
            t0 = time.perf_counter()
            try:
                url = f"{self.config.challenger_url.rstrip('/')}/recommend"
                resp = sync_client.post(url, json=payload)
                t1 = time.perf_counter()
                SHADOW_LATENCY.labels(model_version="challenger-shadow").observe(t1 - t0)
                if resp.status_code >= 400:
                    SHADOW_ERRORS.labels(model_version="challenger-shadow").inc()
            except Exception as e:
                SHADOW_ERRORS.labels(model_version="challenger-shadow").inc()
                logger.debug(f"Shadow call failed: {e}")
            finally:
                self.shadow_queue.task_done()
        sync_client.close()

    def record_canary_call(self, is_error: bool, latency_ms: float):
        """
        Records canary request in sliding window and checks rollback triggers.
        """
        with self._lock:
            self._canary_window.append((time.time(), is_error, latency_ms))
            self._check_rollback_locked()

    def _check_rollback_locked(self):
        if self.rollback_triggered:
            return
        if len(self._canary_window) < self.config.min_sample_size:
            return

        total = len(self._canary_window)
        errors = sum(1 for _, err, _ in self._canary_window if err)
        error_rate = errors / total
        latencies = [lat for _, _, lat in self._canary_window]
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        if error_rate > self.config.max_error_rate:
            self.rollback_triggered = True
            self.rollback_reason = f"Error rate {error_rate * 100:.2f}% exceeded threshold {self.config.max_error_rate * 100:.2f}%"
            self.config.canary_weight = 0
            logger.critical(f"AUTOMATIC ROLLBACK TRIGGERED: {self.rollback_reason}. Canary weight set to 0.")
        elif p95_lat > self.config.max_p95_latency_ms:
            self.rollback_triggered = True
            self.rollback_reason = f"p95 latency {p95_lat:.2f}ms exceeded threshold {self.config.max_p95_latency_ms:.2f}ms"
            self.config.canary_weight = 0
            logger.critical(f"AUTOMATIC ROLLBACK TRIGGERED: {self.rollback_reason}. Canary weight set to 0.")

    def get_status(self) -> RouterStatus:
        with self._lock:
            total = len(self._canary_window)
            errors = sum(1 for _, err, _ in self._canary_window if err)
            err_rate = (errors / total) if total > 0 else 0.0
            latencies = [lat for _, _, lat in self._canary_window]
            p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

            return RouterStatus(
                canary_weight=self.config.canary_weight,
                shadow_enabled=self.config.shadow_enabled,
                rollback_triggered=self.rollback_triggered,
                rollback_reason=self.rollback_reason,
                total_champion_requests=self.total_champion_requests,
                total_canary_requests=self.total_canary_requests,
                total_shadow_requests=self.total_shadow_requests,
                total_shadow_drops=self.total_shadow_drops,
                canary_recent_errors=errors,
                canary_recent_requests=total,
                canary_recent_error_rate=err_rate,
                canary_recent_p95_latency_ms=p95_lat
            )

# Router Application Factory

def create_router_app(router: Optional[TrafficRouter] = None) -> FastAPI:
    r = router or TrafficRouter()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        r.http_client = httpx.AsyncClient(timeout=2.0)
        r.start_shadow_worker()
        yield
        r.stop_shadow_worker()
        if r.http_client:
            await r.http_client.aclose()

    app = FastAPI(title="ShopRank Traffic Router", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "alive", "canary_weight": r.config.canary_weight}

    @app.get("/routing/config", response_model=RouterConfig)
    def get_config():
        return r.config

    @app.post("/routing/config", response_model=RouterConfig)
    def update_config(new_config: RouterConfig):
        r.config = new_config
        if new_config.canary_weight > 0 and r.rollback_triggered:
            # Manual re-enable clears previous rollback flag
            r.rollback_triggered = False
            r.rollback_reason = None
        logger.info(f"Router config updated: canary_weight={new_config.canary_weight}, shadow={new_config.shadow_enabled}")
        return r.config

    @app.get("/routing/status", response_model=RouterStatus)
    def get_status():
        return r.get_status()

    @app.post("/recommend", response_model=RecommendResponse)
    async def recommend(req: RecommendRequest):
        payload = req.model_dump()
        target_canary = is_canary_user(req.user_id, r.config.canary_weight, r.config.canary_salt)

        if target_canary:
            r.total_canary_requests += 1
            t0 = time.perf_counter()
            target_url = f"{r.config.challenger_url.rstrip('/')}/recommend"
            try:
                resp = await r.http_client.post(target_url, json=payload)
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0
                if resp.status_code == 200:
                    r.record_canary_call(is_error=False, latency_ms=lat_ms)
                    return resp.json()
                else:
                    r.record_canary_call(is_error=True, latency_ms=lat_ms)
                    logger.warning(f"Challenger returned {resp.status_code}. Falling back to champion.")
            except Exception as e:
                t1 = time.perf_counter()
                r.record_canary_call(is_error=True, latency_ms=(t1 - t0) * 1000.0)
                logger.warning(f"Challenger exception {e}. Falling back to champion.")
                
            # Canary fallback to champion
            champ_url = f"{r.config.champion_url.rstrip('/')}/recommend"
            champ_resp = await r.http_client.post(champ_url, json=payload)
            if champ_resp.status_code != 200:
                raise HTTPException(status_code=champ_resp.status_code, detail="Both challenger and champion failed")
            return champ_resp.json()

        # Primary route: Champion
        r.total_champion_requests += 1
        champ_url = f"{r.config.champion_url.rstrip('/')}/recommend"
        try:
            resp = await r.http_client.post(champ_url, json=payload)
        except Exception as e:
            logger.error(f"Champion service unreachable: {e}")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Champion unreachable: {e}")

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        # Enqueue non-blocking shadow inference to challenger
        if r.config.shadow_enabled:
            try:
                r.shadow_queue.put_nowait(payload)
            except queue.Full:
                r.total_shadow_drops += 1
                logger.warning("Shadow queue full. Dropped shadow inference request.")

        return resp.json()

    @app.post("/feedback")
    async def forward_feedback(payload: Dict[str, Any]):
        champ_url = f"{r.config.champion_url.rstrip('/')}/feedback"
        try:
            await r.http_client.post(champ_url, json=payload)
        except Exception as e:
            logger.warning(f"Failed to forward feedback to champion: {e}")

        if r.config.challenger_url:
            chal_url = f"{r.config.challenger_url.rstrip('/')}/feedback"
            try:
                await r.http_client.post(chal_url, json=payload)
            except Exception:
                pass
        return {"status": "forwarded", "payload": payload}

    @app.get("/metrics")
    def metrics():
        return Response(content=get_metrics_content(), media_type=CONTENT_TYPE_LATEST)

    return app

# Top-level instance for ASGI runner
router_app = create_router_app()
