import sys
import os

# 🔹 Ajouter le dossier scripts au path pour que Python trouve tes fichiers
sys.path.append("/opt/airflow/scripts")

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Importer les fonctions ETL
from extract_weather import fetch_weather
from extract_pollution import fetch_pollution
from merge_and_load import merge_and_save

# Charger les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv("/opt/airflow/.env")  # chemin dans le container Docker

# Paramètres par défaut du DAG
default_args = {
    'owner': 'Eya',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Créer le DAG
with DAG(
    dag_id='eco_traffic_etl',
    default_args=default_args,
    description='ETL météo + pollution vers MongoDB Atlas',
    schedule_interval='@hourly',  # toutes les heures
    start_date=datetime(2026, 1, 12),
    catchup=False,
    tags=['ETL', 'EcoTraffic'],
) as dag:

    # Fonction qui sera exécutée par Airflow
    def run_etl():
        print("🔹 ETL démarré")
        weather = fetch_weather()
        pollution = fetch_pollution()
        merge_and_save(weather, pollution)
        print("✅ ETL terminé")

    # Tâche Python dans Airflow
    etl_task = PythonOperator(
        task_id='run_etl_pipeline',
        python_callable=run_etl
    )

    etl_task
