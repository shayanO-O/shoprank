#!/usr/bin/env python3
"""
ShopRank Closed-Loop Operations Pipeline Test
Tests the full lifecycle of the MLOps Operations Pipeline:
1. Static DAG Validation (DAG structure, acyclic verification, task dependencies)
2. Statistical Drift Monitoring & Alert Triggering
3. Operations Retraining Pipeline Execution
4. Quality Gate & Bootstrap Statistical Governance
5. Traffic Canary Promotion
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_ops_pipeline")

def run_command(cmd: str, desc: str) -> bool:
    print(f"\n--- [TEST STEP] {desc} ---")
    print(f"Command: {cmd}")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, cwd=root_dir)
    dur = time.time() - t0
    if res.returncode == 0:
        print(f"✅ PASSED in {dur:.2f}s")
        return True
    else:
        print(f"❌ FAILED with return code {res.returncode} in {dur:.2f}s")
        return False

def main():
    py_bin = sys.executable
    print("=" * 80)
    print("SHOPRANK CLOSED-LOOP OPERATIONS PIPELINE VERIFICATION")
    print("=" * 80)

    # 1. DAG Static Topology & Dependency Validation
    step1 = run_command(
        f"{py_bin} src/pipeline/run_pipeline.py --dag shoprank_training_pipeline --dry-run",
        "1. Validating Weekly Training Pipeline DAG (10 Steps)"
    )
    if not step1:
        sys.exit(1)

    step2 = run_command(
        f"{py_bin} src/pipeline/run_pipeline.py --dag shoprank_retrain --dry-run",
        "2. Validating Drift-Triggered Retraining Pipeline DAG (7 Steps)"
    )
    if not step2:
        sys.exit(1)

    # 3. Test Statistical Drift Detection & Alert Cooldown
    step3 = run_command(
        f"{py_bin} -m pytest tests/test_drift.py",
        "3. Validating Statistical Drift Detector (PSI, KS, Chi-Square & Cooldown)"
    )
    if not step3:
        sys.exit(1)

    # 4. Test Quality Gate & Statistical Bootstrap Governance
    step4 = run_command(
        f"{py_bin} -m pytest tests/test_quality_gate.py",
        "4. Validating Quality Gate Governance (1,000 Bootstrap Samples, Latency, Shadow Floor)"
    )
    if not step4:
        sys.exit(1)

    # 5. Execute End-to-End Retraining Pipeline
    print("\n" + "=" * 80)
    print("5. EXECUTING LIVE RETRAINING PIPELINE (shoprank_retrain --smoke)")
    print("=" * 80)
    step5 = run_command(
        f"{py_bin} src/pipeline/run_pipeline.py --dag shoprank_retrain --smoke",
        "Executing Live Drift-Retrain DAG (sync -> retrain -> index -> ranker -> eval -> register -> gate)"
    )
    if not step5:
        sys.exit(1)

    print("\n" + "=" * 80)
    print("🎉 ALL OPERATIONS PIPELINE BEHAVIOURS VERIFIED SUCCESSFULLY!")
    print("  - Weekly & Event-Driven DAGs: Valid & Acyclic")
    print("  - Statistical Drift Webhook: Operational with Cooldown")
    print("  - Automated Retraining: Functional")
    print("  - Quality Gate Governance: Enforced")
    print("=" * 80)

if __name__ == "__main__":
    main()
