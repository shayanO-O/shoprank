from typing import Optional
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Prometheus Metrics Definitions

REQUEST_COUNT = Counter(
    "shoprank_recommendation_requests_total",
    "Total count of recommendation requests processed",
    ["model_version", "status"]
)

REQUEST_LATENCY = Histogram(
    "shoprank_recommendation_latency_seconds",
    "End-to-end request latency for recommendation in seconds",
    ["model_version"],
    buckets=[0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500, 1.0]
)

CANDIDATES_COUNT = Histogram(
    "shoprank_retrieved_candidates_count",
    "Number of candidate items retrieved from retrieval stage",
    ["model_version"],
    buckets=[10, 50, 100, 150, 200, 250, 300]
)

RECOMMENDATIONS_COUNT = Histogram(
    "shoprank_final_recommendations_count",
    "Number of recommendations returned to user after business rules",
    ["model_version"],
    buckets=[1, 5, 10, 20, 50]
)

CACHE_HITS = Counter(
    "shoprank_cache_hits_total",
    "Cache hit counter for feature store lookups",
    ["cache_type"]
)

SHADOW_LATENCY = Histogram(
    "shoprank_shadow_inference_latency_seconds",
    "Inference latency of asynchronous shadow worker",
    ["model_version"],
    buckets=[0.005, 0.010, 0.020, 0.050, 0.100, 0.200, 0.500]
)

SHADOW_ERRORS = Counter(
    "shoprank_shadow_errors_total",
    "Total count of shadow inference worker errors",
    ["model_version"]
)

def record_recommendation_metrics(
    model_version: str,
    status: str,
    latency_seconds: float,
    candidates_count: int,
    final_count: int
):
    """
    Helper to record all standard serving metrics for a request.
    """
    REQUEST_COUNT.labels(model_version=model_version, status=status).inc()
    REQUEST_LATENCY.labels(model_version=model_version).observe(latency_seconds)
    CANDIDATES_COUNT.labels(model_version=model_version).observe(candidates_count)
    RECOMMENDATIONS_COUNT.labels(model_version=model_version).observe(final_count)

def get_metrics_content() -> bytes:
    """
    Generates Prometheus plaintext representation of current metrics.
    """
    return generate_latest()
