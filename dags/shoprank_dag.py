import os
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.python import PythonOperator
except ImportError:
    # Lightweight mock classes for local static testing outside Airflow container
    class DAG:
        def __init__(self, dag_id, *args, **kwargs):
            self.dag_id = dag_id
            self.tasks = {}
            self.tags = kwargs.get("tags", [])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    class BaseOperator:
        def __init__(self, task_id, dag=None, **kwargs):
            self.task_id = task_id
            self.upstream = set()
            self.downstream = set()
            if dag:
                dag.tasks[task_id] = self

        def __rshift__(self, other):
            if isinstance(other, list):
                for item in other:
                    self.downstream.add(item)
                    item.upstream.add(self)
            else:
                self.downstream.add(other)
                other.upstream.add(self)
            return other

        def __lshift__(self, other):
            if isinstance(other, list):
                for item in other:
                    self.upstream.add(item)
                    item.downstream.add(self)
            else:
                self.upstream.add(other)
                other.downstream.add(self)
            return other

    class BashOperator(BaseOperator):
        def __init__(self, task_id, bash_command, **kwargs):
            super().__init__(task_id, **kwargs)
            self.bash_command = bash_command

    class PythonOperator(BaseOperator):
        def __init__(self, task_id, python_callable, **kwargs):
            super().__init__(task_id, **kwargs)
            self.python_callable = python_callable

default_args = {
    "owner": "shoprank-mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="shoprank_training_pipeline",
    default_args=default_args,
    description="End-to-end two-stage recommender training, indexing, evaluation, registration, and quality gate",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["recsys", "mlops", "shoprank"]
) as dag:

    # 1. Validation & Data Quality Checks
    validate = BashOperator(
        task_id="validate_raw_data",
        bash_command="python src/validate.py",
        dag=dag
    )

    # 2. Preprocessing & Filtering
    preprocess = BashOperator(
        task_id="preprocess_interactions",
        bash_command="python src/preprocess.py",
        dag=dag
    )

    # 3. Temporal Stratified Splitting
    split = BashOperator(
        task_id="split_dataset",
        bash_command="python src/split.py",
        dag=dag
    )

    # 4. Baselines Evaluation
    baselines = BashOperator(
        task_id="evaluate_baselines",
        bash_command="python src/baselines.py",
        dag=dag
    )

    # 5. Stage 1 Retrieval Training
    train_retrieval = BashOperator(
        task_id="train_two_tower_retrieval",
        bash_command="python src/two_tower.py",
        dag=dag
    )

    # 6. FAISS Index Construction
    build_index = BashOperator(
        task_id="build_faiss_index",
        bash_command="python src/build_index.py",
        dag=dag
    )

    # 7. Stage 2 LightGBM LambdaRank Training
    train_ranker = BashOperator(
        task_id="train_lambdarank_ranker",
        bash_command="python src/ranker.py",
        dag=dag
    )

    # 8. Evaluation & Latency Benchmarking
    evaluate = BashOperator(
        task_id="evaluate_system",
        bash_command="python src/evaluate.py",
        dag=dag
    )

    # 9. Atomic Model Registration with Checksum Manifest
    register = BashOperator(
        task_id="register_model_bundle",
        bash_command="python src/register.py",
        dag=dag
    )

    # 10. Automated Quality Gate & Promotion
    quality_gate = BashOperator(
        task_id="evaluate_quality_gate",
        bash_command="python src/pipeline/quality_gate.py --challenger-results metrics/results.json",
        dag=dag
    )

    # Explicit Pipeline Task Dependencies
    validate >> preprocess >> split
    split >> baselines
    split >> train_retrieval >> build_index >> train_ranker >> evaluate >> register >> quality_gate
