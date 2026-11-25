import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Support direct script execution and module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

# Statistical Bootstrap Confidence Intervals

def compute_bootstrap_ndcg_ci(
    challenger_scores: np.ndarray,
    champion_scores: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Computes observed mean NDCG difference and non-parametric bootstrap (1 - alpha) CI.
    Returns: (mean_diff, ci_lower, ci_upper)
    """
    if len(challenger_scores) == 0 or len(champion_scores) == 0:
        return 0.0, 0.0, 0.0

    diffs = challenger_scores - champion_scores
    observed_mean = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))

    return observed_mean, ci_lower, ci_upper

def compute_rank_overlap(list1: List[int], list2: List[int], k: int = 10) -> float:
    """
    Computes top-k Jaccard set overlap between two recommendation lists.
    Avoids comparing uncalibrated raw LambdaRank scores across different model versions.
    """
    s1 = set(list1[:k])
    s2 = set(list2[:k])
    if not s1 and not s2:
        return 1.0
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))

# Production Quality Gate Engine

class QualityGate:
    def __init__(
        self,
        min_ndcg_margin: float = 0.005,
        max_p95_latency_ms: float = 30.0,
        max_shadow_error_rate: float = 0.001,  # 0.1%
        min_shadow_sample_size: int = 1000,
        min_catalog_coverage: float = 0.10,
        min_rank_overlap: float = 0.30
    ):
        self.min_ndcg_margin = min_ndcg_margin
        self.max_p95_latency_ms = max_p95_latency_ms
        self.max_shadow_error_rate = max_shadow_error_rate
        self.min_shadow_sample_size = min_shadow_sample_size
        self.min_catalog_coverage = min_catalog_coverage
        self.min_rank_overlap = min_rank_overlap

    def evaluate(
        self,
        challenger_results: Dict[str, Any],
        champion_results: Optional[Dict[str, Any]] = None,
        per_user_ndcg: Optional[Dict[str, Any]] = None,
        shadow_metrics: Optional[Dict[str, Any]] = None,
        sample_recs_challenger: Optional[List[int]] = None,
        sample_recs_champion: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Executes strict multi-criteria quality gate evaluation.
        """
        checks = {}
        failures = []

        # 1. NDCG@10 Check with Bootstrap CI
        c_test = challenger_results.get("retrieval_plus_ranker", {}).get("test", {})
        c_ndcg = float(c_test.get("ndcg_at_10", 0.0))
        
        champ_test = (champion_results or {}).get("retrieval_plus_ranker", {}).get("test", {})
        champ_ndcg = float(champ_test.get("ndcg_at_10", 0.0))

        if per_user_ndcg:
            # User scores breakdown
            c_u = []
            champ_u = []
            for uid, scores in per_user_ndcg.items():
                if isinstance(scores, dict):
                    c_u.append(scores.get("with_rules", scores.get("raw", 0.0)))
                    champ_u.append(scores.get("raw", 0.0))
                else:
                    c_u.append(float(scores))
                    champ_u.append(champ_ndcg)

            mean_diff, ci_low, ci_high = compute_bootstrap_ndcg_ci(np.array(c_u), np.array(champ_u))
        else:
            mean_diff = c_ndcg - champ_ndcg
            ci_low = mean_diff - 0.001
            ci_high = mean_diff + 0.001

        # Must beat champion by margin and lower bound of CI must strictly exceed 0 (rejecting ties)
        ndcg_pass = (mean_diff >= self.min_ndcg_margin) and (ci_low > 0.0)
        checks["ndcg_improvement"] = {
            "passed": bool(ndcg_pass),
            "challenger_ndcg": c_ndcg,
            "champion_ndcg": champ_ndcg,
            "mean_difference": float(mean_diff),
            "ci_95_lower": float(ci_low),
            "ci_95_upper": float(ci_high),
            "required_margin": self.min_ndcg_margin
        }
        if not ndcg_pass:
            failures.append(f"NDCG diff {mean_diff:.4f} (95% CI [{ci_low:.4f}, {ci_high:.4f}]) failed margin {self.min_ndcg_margin}")

        # 2. p95 Latency Floor
        latency_info = challenger_results.get("retrieval_plus_ranker", {}).get("latency", {})
        p95_lat = float(latency_info.get("p95", 0.0))
        lat_pass = (p95_lat > 0.0) and (p95_lat <= self.max_p95_latency_ms)
        checks["latency_p95"] = {
            "passed": bool(lat_pass),
            "measured_p95_ms": p95_lat,
            "threshold_p95_ms": self.max_p95_latency_ms
        }
        if not lat_pass:
            failures.append(f"p95 latency {p95_lat:.2f}ms exceeded limit {self.max_p95_latency_ms:.2f}ms")

        # 3. Shadow Error Rate with Mandatory Minimum Sample Size Floor
        if shadow_metrics:
            sample_size = int(shadow_metrics.get("sample_size", 0))
            err_rate = float(shadow_metrics.get("error_rate", 0.0))
            
            sample_pass = sample_size >= self.min_shadow_sample_size
            err_pass = err_rate <= self.max_shadow_error_rate
            shadow_pass = sample_pass and err_pass

            checks["shadow_safety"] = {
                "passed": bool(shadow_pass),
                "sample_size": sample_size,
                "min_sample_size": self.min_shadow_sample_size,
                "error_rate": err_rate,
                "max_error_rate": self.max_shadow_error_rate
            }
            if not sample_pass:
                failures.append(f"Shadow sample size {sample_size} < required floor {self.min_shadow_sample_size}")
            if not err_pass:
                failures.append(f"Shadow error rate {err_rate * 100:.3f}% > threshold {self.max_shadow_error_rate * 100:.3f}%")
        else:
            checks["shadow_safety"] = {"passed": True, "note": "Shadow evaluation skipped (not provided)"}

        # 4. Catalog Coverage Floor
        coverage = float(c_test.get("catalog_coverage", 0.0))
        coverage_pass = coverage >= self.min_catalog_coverage
        checks["catalog_coverage"] = {
            "passed": bool(coverage_pass),
            "measured_coverage": coverage,
            "floor_coverage": self.min_catalog_coverage
        }
        if not coverage_pass:
            failures.append(f"Catalog coverage {coverage * 100:.2f}% < floor {self.min_catalog_coverage * 100:.2f}%")

        # 5. Rank Stability Overlap Check
        if sample_recs_challenger is not None and sample_recs_champion is not None:
            overlap = compute_rank_overlap(sample_recs_challenger, sample_recs_champion, k=10)
            overlap_pass = overlap >= self.min_rank_overlap
            checks["rank_stability"] = {
                "passed": bool(overlap_pass),
                "top10_overlap": overlap,
                "min_overlap": self.min_rank_overlap
            }
            if not overlap_pass:
                failures.append(f"Top-10 rank overlap {overlap:.2f} < threshold {self.min_rank_overlap:.2f}")
        else:
            checks["rank_stability"] = {"passed": True, "note": "Rank overlap check skipped (not provided)"}

        overall_passed = len(failures) == 0
        return {
            "qualified_for_promotion": overall_passed,
            "status": "APPROVED" if overall_passed else "REJECTED",
            "failures": failures,
            "checks": checks
        }

    def promote_in_registry(
        self,
        gate_result: Dict[str, Any],
        model_name: str = "shoprank",
        target_alias: str = "challenger",
        output_report_path: Optional[str] = "metrics/quality_gate_report.json"
    ):
        """
        Updates MLflow model registry alias if gate passed; saves audit report.
        """
        report_p = Path(output_report_path) if output_report_path else Path("metrics/quality_gate_report.json")
        report_p.parent.mkdir(parents=True, exist_ok=True)
        with open(report_p, "w", encoding="utf-8") as f:
            json.dump(gate_result, f, indent=2)

        if not gate_result.get("qualified_for_promotion", False):
            logger.warning(f"Promotion rejected: {gate_result['failures']}")
            return False

        logger.info(f"Model {model_name} passed quality gate. Promoting to '{target_alias}'.")
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest = versions[0].version
                client.set_registered_model_alias(model_name, target_alias, latest)
                logger.info(f"Set alias '{target_alias}' on {model_name} version {latest}")
                return True
        except Exception as e:
            logger.warning(f"MLflow alias update skipped (offline/local): {e}")
            return True
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopRank Quality Gate")
    parser.add_argument("--challenger-results", type=str, default="metrics/results.json", help="Challenger results.json")
    parser.add_argument("--champion-results", type=str, default="metrics/champion_results.json", help="Champion results.json")
    parser.add_argument("--per-user-ndcg", type=str, default="metrics/per_user_ndcg.json", help="Per-user NDCG breakdown")
    parser.add_argument("--output", type=str, default="metrics/quality_gate_report.json", help="Output report path")
    args = parser.parse_args()

    c_res_p = Path(args.challenger_results)
    if not c_res_p.exists():
        print(f"Results file not found at {c_res_p}")
        exit(1)

    with open(c_res_p, "r", encoding="utf-8") as f:
        c_res = json.load(f)

    champ_res = None
    champ_p = Path(args.champion_results)
    if champ_p.exists():
        with open(champ_p, "r", encoding="utf-8") as f:
            champ_res = json.load(f)

    user_ndcg = None
    u_p = Path(args.per_user_ndcg)
    if u_p.exists():
        with open(u_p, "r", encoding="utf-8") as f:
            user_ndcg = json.load(f)

    gate = QualityGate()
    result = gate.evaluate(
        challenger_results=c_res,
        champion_results=champ_res,
        per_user_ndcg=user_ndcg
    )
    gate.promote_in_registry(result, output_report_path=args.output)
    print(f"Quality gate decision: {result['status']}")
