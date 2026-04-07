# Airflow Integration

Cette intégration ajoute Airflow au-dessus du flux actuel :

1. exécuter `load_to_mongodb.py`
2. lancer `kedro run --pipeline data_processing`
3. vérifier que `traffic_cleaned.csv` a bien été généré

## Fichiers

- `airflow/dags/ecotraffic_ingestion_preprocessing.py` : la DAG principale
- `airflow/Dockerfile` : image Airflow avec les dépendances du projet
- `docker-compose.airflow.yml` : démarrage local via Docker

## Démarrage local

Depuis la racine du dépôt :

```powershell
docker compose -f docker-compose.airflow.yml up --build
```

Puis ouvre :

```text
http://localhost:8080
```

Avec `airflow standalone`, les identifiants générés au premier démarrage apparaissent dans les logs :

```powershell
docker compose -f docker-compose.airflow.yml logs airflow
```

## Ce que la DAG orchestre aujourd'hui

- ingestion API Rennes -> MongoDB
- preprocessing Kedro depuis MongoDB

## Extension future

Tu pourras ajouter ensuite d'autres tâches Airflow dans cette même DAG :

- fusion avec les données de ton amie
- upload vers PostgreSQL / Supabase
- entraînement ML
- déploiement cloud
