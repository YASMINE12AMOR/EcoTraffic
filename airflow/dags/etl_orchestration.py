from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task

PROJECT_ROOT = Path("/opt/airflow")
KEDRO_PROJECT_ROOT = PROJECT_ROOT / "kedro_preprocessing"


def _build_env() -> dict[str, str]:
    env = os.environ.copy()

    pythonpath_entries = [
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "app"),
        str(KEDRO_PROJECT_ROOT / "src"),
    ]

    current_pythonpath = env.get("PYTHONPATH")
    if current_pythonpath:
        pythonpath_entries.append(current_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return env


def _run_command(command: list[str], cwd: Path) -> None:
    subprocess.run(
        command,
        cwd=str(cwd),
        env=_build_env(),
        check=True,
    )


@dag(
    dag_id="ecotraffic_full_pipeline",
    description="API -> MongoDB -> Kedro -> PostgreSQL",
    start_date=datetime(2026, 4, 6),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "eya",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["ecotraffic", "etl", "mongodb", "kedro", "postgresql"],
)
def ecotraffic_full_pipeline():
    @task(task_id="extract_to_mongo")
    def extract_to_mongo() -> None:
        _run_command(["python", "-m", "app.pipeline"], cwd=PROJECT_ROOT)

    @task(task_id="transform_with_kedro", retries=3, retry_delay=timedelta(seconds=30))
    def transform_with_kedro() -> None:
        import time
        time.sleep(5)
        _run_command(["kedro", "run"], cwd=KEDRO_PROJECT_ROOT)

    @task(task_id="check_postgres_table")
    def check_postgres_table() -> str:
        import os
        from sqlalchemy import create_engine, text

        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI manquant dans les variables d'environnement")

        engine = create_engine(postgres_uri)

        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM environment_features"))
            count = result.scalar()

        if count is None or count == 0:
            raise ValueError("La table environment_features est vide ou introuvable")

        print(f"Nombre de lignes dans environment_features: {count}")
        return f"Nombre de lignes dans environment_features: {count}"

    mongo_step = extract_to_mongo()
    kedro_step = transform_with_kedro()
    postgres_check = check_postgres_table()

    mongo_step >> kedro_step >> postgres_check


ecotraffic_full_pipeline()