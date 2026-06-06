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
    schedule="@hourly",
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
        _run_command(["python", "load_to_mongodb.py"], cwd=KEDRO_PROJECT_ROOT)

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

    @task(task_id="monitor_data_quality")
    def monitor_data_quality() -> str:
        import json
        import pandas as pd

        df = pd.read_csv(str(OUTPUT_FILE))
        VALID_STATUTS = {"freeflow", "heavy", "congested", "unknown"}

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_rows": len(df),
            "null_route": int(df["Route"].isna().sum()),
            "null_speed": int(df["Vitesse Moyenne (km/h)"].isna().sum()),
            "null_date": int(df["Date/Heure"].isna().sum()),
            "speed_out_of_range": int(((df["Vitesse Moyenne (km/h)"] < 0) | (df["Vitesse Moyenne (km/h)"] > 130)).sum()),
            "invalid_statuses": list(set(df["Statut Trafic"].dropna().unique()) - VALID_STATUTS),
            "duplicates": int(df.duplicated().sum()),
        }

        log_path = Path("/opt/airflow/logs/monitor_traffic_quality.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(report, f, indent=2)

        if report["total_rows"] == 0:
            raise ValueError(f"MONITOR: Aucune ligne dans traffic_cleaned.csv")
        if report["null_speed"] > 0:
            raise ValueError(f"MONITOR: {report['null_speed']} vitesses nulles détectées")
        if report["speed_out_of_range"] > 0:
            raise ValueError(f"MONITOR: {report['speed_out_of_range']} vitesses hors plage [0-130]")
        if report["invalid_statuses"]:
            raise ValueError(f"MONITOR: Statuts invalides détectés : {report['invalid_statuses']}")

        print(f"Data quality OK:\n{json.dumps(report, indent=2)}")
        return json.dumps(report)

    @task(task_id="load_traffic_to_postgres", retries=2, retry_delay=timedelta(minutes=1))
    def load_traffic_to_postgres() -> str:
        import os
        import pandas as pd
        from sqlalchemy import create_engine

        postgres_uri = os.getenv("POSTGRES_URI")
        if not postgres_uri:
            raise ValueError("POSTGRES_URI manquant dans les variables d'environnement")

        df = pd.read_csv(str(OUTPUT_FILE))
        if df.empty:
            print("traffic_cleaned.csv vide, rien à insérer.")
            return "empty"

        df["Date/Heure"] = pd.to_datetime(df["Date/Heure"], utc=True, errors="coerce")
        for col in ["Vitesse Moyenne (km/h)", "Vitesse Max (km/h)", "Temps de Trajet (s)", "Fiabilité (%)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Statut Trafic" in df.columns:
            df["Statut Trafic"] = df["Statut Trafic"].astype(str).str.lower().str.strip()

        table = os.getenv("POSTGRES_TABLE_TRAFFIC", "traffic_features")
        engine = create_engine(postgres_uri)
        df.to_sql(table, engine, if_exists="append", index=False)

        print(f"{len(df)} lignes insérées dans PostgreSQL -> table {table}")
        return f"{len(df)} lignes insérées dans {table}"

    raw_ingestion = load_raw_data_into_mongodb()
    kedro_preprocessing = run_kedro_preprocessing()
    output_validation = validate_kedro_output()
    quality_check = monitor_data_quality()
    traffic_to_pg = load_traffic_to_postgres()

    raw_ingestion >> kedro_preprocessing >> output_validation >> quality_check >> traffic_to_pg


ecotraffic_ingestion_preprocessing()
