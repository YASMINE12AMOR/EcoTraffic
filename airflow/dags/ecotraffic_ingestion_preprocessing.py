from __future__ import annotations

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task  # type: ignore[reportMissingImports]

PROJECT_ROOT = Path("/opt/ecotraffic")
KEDRO_PROJECT_ROOT = PROJECT_ROOT / "ecotraffic"
OUTPUT_FILE = KEDRO_PROJECT_ROOT / "data" / "03_primary" / "traffic_cleaned.csv"


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_entries = [
        str(PROJECT_ROOT),
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
    dag_id="ecotraffic_ingestion_preprocessing",
    description="Loads raw traffic data into MongoDB, then runs Kedro preprocessing.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "owner": "ecotraffic",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["ecotraffic", "kedro", "airflow", "mongodb"],
)
def ecotraffic_ingestion_preprocessing():
    @task(task_id="load_raw_data_into_mongodb")
    def load_raw_data_into_mongodb() -> None:
        _run_command(["python", "load_to_mongodb.py"], cwd=PROJECT_ROOT)

    @task(task_id="run_kedro_preprocessing")
    def run_kedro_preprocessing() -> None:
        _run_command(
            ["kedro", "run", "--pipeline", "data_processing"],
            cwd=KEDRO_PROJECT_ROOT,
        )

    @task(task_id="validate_kedro_output")
    def validate_kedro_output() -> str:
        if not OUTPUT_FILE.exists():
            raise FileNotFoundError(
                f"Expected Kedro output file at {OUTPUT_FILE}, but it was not created."
            )
        if OUTPUT_FILE.stat().st_size == 0:
            raise ValueError(f"The file {OUTPUT_FILE} exists but is empty.")
        return str(OUTPUT_FILE)

    raw_ingestion = load_raw_data_into_mongodb()
    kedro_preprocessing = run_kedro_preprocessing()
    output_validation = validate_kedro_output()

    raw_ingestion >> kedro_preprocessing >> output_validation


ecotraffic_ingestion_preprocessing()
