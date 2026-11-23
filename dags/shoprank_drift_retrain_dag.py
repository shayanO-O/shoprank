from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:
    from dags.shoprank_dag import DAG, BashOperator

default_args = {
    "owner": "shoprank-monitoring",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="shoprank_retrain",
    default_args=default_args,
    description="Automated retraining pipeline triggered upon drift detector alert",
    schedule_interval=None,  # Webhook / event-driven only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["recsys", "mlops", "drift-retrain"]
) as dag:

    # 1. Pull recent interactions and update reference baseline
    sync_data = BashOperator(
        task_id="sync_production_interactions",
        bash_command="python src/preprocess.py",
        dag=dag
    )

    # 2. Retrain retrieval
    train_retrieval = BashOperator(
        task_id="retrain_retrieval",
        bash_command="python src/two_tower.py",
        dag=dag
    )

    # 3. Rebuild index
    build_index = BashOperator(
        task_id="rebuild_index",
        bash_command="python src/build_index.py",
        dag=dag
    )

    # 4. Retrain ranker
    train_ranker = BashOperator(
        task_id="retrain_ranker",
        bash_command="python src/ranker.py",
        dag=dag
    )

    # 5. Evaluate
    evaluate = BashOperator(
        task_id="evaluate_retrained_model",
        bash_command="python src/evaluate.py",
        dag=dag
    )

    # 6. Register bundle
    register = BashOperator(
        task_id="register_bundle",
        bash_command="python src/register.py",
        dag=dag
    )

    # 7. Quality gate evaluation
    quality_gate = BashOperator(
        task_id="evaluate_gate",
        bash_command="python src/pipeline/quality_gate.py --challenger-results metrics/results.json",
        dag=dag
    )

    sync_data >> train_retrieval >> build_index >> train_ranker >> evaluate >> register >> quality_gate
