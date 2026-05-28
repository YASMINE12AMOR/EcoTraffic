#!/bin/bash
set -e

echo "[EcoTraffic] Initialisation de la base Airflow..."
airflow db migrate

echo "[EcoTraffic] Creation de l'utilisateur admin..."
airflow users create \
  --username admin \
  --password admin123 \
  --firstname EcoTraffic \
  --lastname Admin \
  --role Admin \
  --email admin@ecotraffic.fr \
  2>/dev/null || echo "[EcoTraffic] Utilisateur admin deja existant."

echo "[EcoTraffic] Demarrage Airflow (webserver + scheduler)..."
exec airflow standalone
