#!/usr/bin/env python3
"""
ShopRank Operations Pipeline Runner
Executes Airflow DAGs locally in topological order with step-by-step logging,
timing, dependency verification, and exit status handling.
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Set

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("operations_pipeline")

def get_topological_order(tasks: Dict[str, Any]) -> List[Any]:
    """
    Computes deterministic topological task execution order using Kahn's algorithm.
    """
    in_degree = {task_id: len(task.upstream) for task_id, task in tasks.items()}
    queue = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
    ordered = []

    while queue:
        curr_id = queue.popleft()
        ordered.append(tasks[curr_id])
        for down in tasks[curr_id].downstream:
            in_degree[down.task_id] -= 1
            if in_degree[down.task_id] == 0:
                queue.append(down.task_id)

    if len(ordered) != len(tasks):
        raise ValueError("Cycle detected in DAG task dependencies!")

    return ordered

def run_dag(dag_name: str, dry_run: bool = False, smoke: bool = False, verbose: bool = True) -> bool:
    """
    Loads and executes tasks in the given DAG.
    """
    if dag_name == "shoprank_training_pipeline":
        from dags.shoprank_dag import dag
    elif dag_name in ("shoprank_retrain", "drift_retrain"):
        from dags.shoprank_drift_retrain_dag import dag
    else:
        raise ValueError(f"Unknown DAG: {dag_name}. Available: shoprank_training_pipeline, shoprank_retrain")

    ordered_tasks = get_topological_order(dag.tasks)
    mode_str = "DRY RUN" if dry_run else ("EXECUTE (SMOKE)" if smoke else "EXECUTE (FULL)")
    
    print("\n" + "=" * 75)
    print(f"OPERATIONS PIPELINE: {dag.dag_id.upper()}")
    print(f"Description: {getattr(dag, 'description', '')}")
    print(f"Total Steps: {len(ordered_tasks)} | Mode: {mode_str}")
    print("=" * 75)

    for i, t in enumerate(ordered_tasks, 1):
        upstream_ids = [u.task_id for u in t.upstream]
        cmd = getattr(t, "bash_command", "python_callable")
        print(f"  Step {i}/{len(ordered_tasks)}: [{t.task_id}]")
        print(f"    Depends on: {upstream_ids if upstream_ids else 'None (Root)'}")
        print(f"    Command:    {cmd}")

    if dry_run:
        print("\n[Dry Run] All DAG dependencies validated. No commands executed.")
        return True

    print("\nStarting pipeline execution...")
    pipeline_t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir)

    for i, t in enumerate(ordered_tasks, 1):
        cmd = getattr(t, "bash_command", None)
        if not cmd:
            print(f"\nSkipping task {t.task_id}: no bash command defined.")
            continue

        # Replace 'python ' with current environment's Python executable
        if "python " in cmd:
            formatted_cmd = cmd.replace("python ", f"{sys.executable} ")
        else:
            formatted_cmd = cmd

        if smoke:
            if any(target in formatted_cmd for target in [
                "src/preprocess.py", "src/split.py", "src/two_tower.py",
                "src/build_index.py", "src/ranker.py", "src/evaluate.py", "src/register.py"
            ]):
                if "--smoke" not in formatted_cmd:
                    formatted_cmd += " --smoke"
            if "src/pipeline/quality_gate.py" in formatted_cmd and "metrics/results.json" in formatted_cmd:
                formatted_cmd = formatted_cmd.replace("metrics/results.json", "scratch/smoke/metrics/results.json")

        print("\n" + "-" * 75)
        print(f"[{i}/{len(ordered_tasks)}] Running: {t.task_id}")
        print(f"Command: {formatted_cmd}")
        print("-" * 75)

        step_t0 = time.time()
        ret = subprocess.run(formatted_cmd, shell=True, cwd=root_dir, env=env)
        step_dur = time.time() - step_t0

        if ret.returncode != 0:
            print(f"\n❌ Task '{t.task_id}' FAILED with exit code {ret.returncode} after {step_dur:.2f}s!")
            print(f"Operations pipeline ABORTED at step {i}/{len(ordered_tasks)}.")
            return False

        print(f"✅ Task '{t.task_id}' completed successfully in {step_dur:.2f}s")

    total_dur = time.time() - pipeline_t0
    print("\n" + "=" * 75)
    print(f"🎉 OPERATIONS PIPELINE [{dag.dag_id}] COMPLETED SUCCESSFULLY in {total_dur:.2f}s!")
    print("=" * 75)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShopRank Operations Pipeline Runner")
    parser.add_argument("--dag", type=str, default="shoprank_retrain", choices=["shoprank_training_pipeline", "shoprank_retrain"], help="DAG to execute")
    parser.add_argument("--dry-run", action="store_true", help="Validate DAG dependencies without executing commands")
    parser.add_argument("--smoke", action="store_true", help="Execute in fast smoke mode (ideal for rapid integration tests)")
    args = parser.parse_args()

    success = run_dag(dag_name=args.dag, dry_run=args.dry_run, smoke=args.smoke)
    sys.exit(0 if success else 1)
