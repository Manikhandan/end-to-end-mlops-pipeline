"""Optional Airflow wrapper. The pipeline is runnable without Airflow."""

from __future__ import annotations

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime

    from mlops_pipeline.pipelines import run_deployment, run_training

    with DAG(
        dag_id="mlops_training_then_deploy",
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["mlops"],
    ) as dag:
        PythonOperator(task_id="training", python_callable=run_training) >> PythonOperator(
            task_id="deployment", python_callable=run_deployment
        )
except ImportError:
    dag = None
