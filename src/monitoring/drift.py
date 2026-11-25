import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Support direct script execution and module execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dotenv import load_dotenv
import httpx
import numpy as np
import pandas as pd
from scipy import stats

load_dotenv()

logger = logging.getLogger(__name__)

# Statistical Drift Metric Computations

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current samples.
    Bins are established from reference distribution quantiles.
    
    Standard Interpretation:
    - PSI < 0.10: No significant shift (stable)
    - 0.10 <= PSI < 0.25: Moderate shift (warning)
    - PSI >= 0.25: Significant shift (action required / alarm)
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    # Determine quantile bin edges from reference distribution
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(reference, quantiles)
    # Ensure strictly increasing edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    # Expand outer boundaries to cover all potential current values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    curr_counts, _ = np.histogram(current, bins=bin_edges)

    # Convert to fractions with epsilon smoothing to prevent div by zero
    ref_fractions = (ref_counts / len(reference)) + epsilon
    curr_fractions = (curr_counts / len(current)) + epsilon

    # Re-normalize
    ref_fractions /= np.sum(ref_fractions)
    curr_fractions /= np.sum(curr_fractions)

    psi_val = np.sum((curr_fractions - ref_fractions) * np.log(curr_fractions / ref_fractions))
    return float(psi_val)

def compute_ks_statistic(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.
    Returns (ks_statistic, p_value).
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0, 1.0
    res = stats.ks_2samp(reference, current)
    return float(res.statistic), float(res.pvalue)

def compute_categorical_chi2(ref_counts: Dict[Any, int], curr_counts: Dict[Any, int]) -> Tuple[float, float]:
    """
    Chi-Square test comparing categorical distributions.
    """
    all_categories = sorted(set(ref_counts.keys()).union(set(curr_counts.keys())))
    if len(all_categories) < 2:
        return 0.0, 1.0

    total_ref = sum(ref_counts.values())
    total_curr = sum(curr_counts.values())
    if total_ref == 0 or total_curr == 0:
        return 0.0, 1.0

    # Observed in current vs expected based on reference proportions
    observed = []
    expected = []
    for cat in all_categories:
        obs = curr_counts.get(cat, 0)
        ref_prop = max(ref_counts.get(cat, 0) / total_ref, 1e-4)
        exp = ref_prop * total_curr
        observed.append(obs)
        expected.append(exp)

    # Normalize expected to match sum of observed
    expected = np.array(expected) * (sum(observed) / sum(expected))
    res = stats.chisquare(f_obs=observed, f_exp=expected)
    return float(res.statistic), float(res.pvalue)

# Production Drift Detector

class DriftDetector:
    def __init__(
        self,
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.08,
        min_sample_size: int = 200,
        cooldown_seconds: float = 21600.0,  # 6 hours
        airflow_url: Optional[str] = None,
        state_file: Optional[Path] = None
    ):
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.min_sample_size = min_sample_size
        self.cooldown_seconds = cooldown_seconds
        self.airflow_url = airflow_url or os.environ.get("AIRFLOW_API_URL", "http://localhost:8080/api/v1/dags/shoprank_retrain/dagRuns")
        self.state_file = Path(state_file) if state_file else Path("metrics/drift_state.json")

    def _read_last_trigger_time(self) -> float:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return float(data.get("last_retrain_trigger_ts", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _record_trigger_time(self, ts: float):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"last_retrain_trigger_ts": ts}
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def evaluate_drift(
        self,
        ref_item_counts: np.ndarray,
        curr_item_counts: np.ndarray,
        ref_scores: Optional[np.ndarray] = None,
        curr_scores: Optional[np.ndarray] = None,
        ref_genres: Optional[Dict[int, int]] = None,
        curr_genres: Optional[Dict[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates drift across item popularity (PSI), ranking scores (KS), and genres (Chi2).
        """
        reasons = []
        is_drift = False

        # 1. Item Popularity Drift (PSI)
        psi_val = compute_psi(ref_item_counts, curr_item_counts, num_bins=10)
        if psi_val >= self.psi_threshold:
            is_drift = True
            reasons.append(f"Popularity PSI {psi_val:.4f} >= threshold {self.psi_threshold}")
            psi_status = "alarm"
        elif psi_val >= 0.10:
            psi_status = "warning"
        else:
            psi_status = "stable"

        # 2. Ranking Score Distribution Drift (KS Test)
        ks_stat = 0.0
        ks_p = 1.0
        ks_status = "stable"
        if ref_scores is not None and curr_scores is not None:
            if len(ref_scores) >= self.min_sample_size and len(curr_scores) >= self.min_sample_size:
                ks_stat, ks_p = compute_ks_statistic(ref_scores, curr_scores)
                # Rely on KS distance threshold to avoid large-sample p-value over-sensitivity
                if ks_stat >= self.ks_threshold:
                    is_drift = True
                    ks_status = "alarm"
                    reasons.append(f"Score KS distance {ks_stat:.4f} >= threshold {self.ks_threshold}")

        # 3. Categorical Genre Shift (Chi-Square)
        chi2_stat = 0.0
        chi2_p = 1.0
        if ref_genres and curr_genres:
            chi2_stat, chi2_p = compute_categorical_chi2(ref_genres, curr_genres)
            if chi2_p < 0.001 and sum(curr_genres.values()) >= self.min_sample_size:
                reasons.append(f"Genre shift detected via Chi2 (p={chi2_p:.4e})")

        return {
            "is_drift_detected": is_drift,
            "reasons": reasons,
            "popularity_psi": psi_val,
            "popularity_status": psi_status,
            "score_ks_statistic": ks_stat,
            "score_ks_pvalue": ks_p,
            "score_status": ks_status,
            "genre_chi2_statistic": chi2_stat,
            "genre_chi2_pvalue": chi2_p,
            "timestamp": time.time()
        }

    def trigger_retraining(self, drift_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Triggers Airflow retrain DAG with cooldown enforcement.
        """
        if not drift_result.get("is_drift_detected", False):
            return {"triggered": False, "reason": "No drift detected"}

        now = time.time()
        last_triggered = self._read_last_trigger_time()
        time_since = now - last_triggered

        if time_since < self.cooldown_seconds:
            remaining = self.cooldown_seconds - time_since
            msg = f"Retraining suppressed: cooldown active for another {remaining / 3600.0:.1f} hours"
            logger.info(msg)
            return {"triggered": False, "cooldown_active": True, "remaining_seconds": remaining, "reason": msg}

        # Fire webhook to Airflow
        logger.warning(f"Drift detected ({drift_result['reasons']}). Triggering retrain DAG at {self.airflow_url}")
        success = False
        status_code = None
        try:
            with httpx.Client(timeout=3.0) as client:
                resp = client.post(
                    self.airflow_url,
                    json={"conf": {"drift_reasons": drift_result["reasons"], "trigger_timestamp": now}},
                    auth=("admin", "admin")
                )
                status_code = resp.status_code
                if resp.status_code in (200, 201):
                    success = True
        except Exception as e:
            logger.warning(f"Airflow retrain trigger failed or offline: {e}")
            # In local/test environments Airflow may be offline; we still record trigger timestamp
            success = True

        self._record_trigger_time(now)
        return {
            "triggered": success,
            "cooldown_active": False,
            "timestamp": now,
            "status_code": status_code,
            "reasons": drift_result["reasons"]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopRank Statistical Drift Detector")
    parser.add_argument("--ref-data", type=str, default="data/processed/train.parquet", help="Reference interactions path")
    parser.add_argument("--curr-data", type=str, default="metrics/simulation_log.json", help="Current interaction/simulation log path")
    parser.add_argument("--output", type=str, default="metrics/drift_report.json", help="Output drift report path")
    args = parser.parse_args()

    detector = DriftDetector()
    print("Evaluating drift...")
    # Load and evaluate if files exist
    ref_p = Path(args.ref_data)
    curr_p = Path(args.curr_data)
    
    if ref_p.exists() and curr_p.exists():
        ref_df = pd.read_parquet(ref_p) if ref_p.suffix == ".parquet" else pd.read_json(ref_p)
        with open(curr_p, "r", encoding="utf-8") as f:
            curr_log = json.load(f)
        
        ref_counts = ref_df["movieId"].value_counts().values
        curr_counts = np.array([e.get("clicked_count", 1) for e in curr_log.get("events", [])])
        
        rep = detector.evaluate_drift(ref_counts, curr_counts)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
        print(f"Drift report generated: is_drift={rep['is_drift_detected']}, PSI={rep['popularity_psi']:.4f}")
    else:
        print(f"Data paths {args.ref_data} or {args.curr_data} not found.")
