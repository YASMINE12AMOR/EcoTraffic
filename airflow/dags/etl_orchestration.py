from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from app.pipeline import run_pipeline

def run_etl():
    count = run_pipeline()
    print(f"Inserted {count} records")

with DAG(
    dag_id="ecotraffic_etl",
    start_date=datetime(2026, 1, 1),
    schedule_interval="@daily",
    catchup=False,
) as dag:

    task = PythonOperator(
        task_id="run_pipeline",
        python_callable=run_etl,
    )