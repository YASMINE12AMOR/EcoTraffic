# EcoTraffic — Pipeline de données & Machine Learning (Rennes)

Projet de Data Engineering et Machine Learning collectant des données de **trafic routier**, de **météo** et de **qualité de l'air** sur la ville de **Rennes**, avec un pipeline ETL automatisé et un modèle ML de prédiction de la qualité de l'air.

---

## Objectifs

- Collecter en temps réel les données trafic, météo et pollution via des APIs
- Nettoyer, transformer et stocker les données dans des bases structurées
- Automatiser l'exécution du pipeline avec Apache Airflow
- Appliquer des modèles ML pour prédire le comportement du trafic et la qualité de l'air
- Combiner les prédictions pour produire un **EcoTraffic Score** par zone de Rennes

---

## Sources de données

| API | Données | Fréquence | Localisation |
|---|---|---|---|
| Rennes Métropole OpenData | Trafic en temps réel (vitesse, statut, routes) | Temps réel | 89 routes de Rennes |
| Open-Meteo Weather | Météo horaire (température, pluie, vent) | Horaire | Rennes (48.1173, -1.6778) |
| Open-Meteo Air Quality | Pollution horaire (PM2.5, PM10, NO2, CO, ozone) | Horaire | Rennes (48.1173, -1.6778) |

---

## Architecture complète

```
┌─────────────────────────────────────────────────────────────────┐
│                         SOURCES DE DONNÉES                      │
│                                                                 │
│  API Trafic Rennes        API Météo           API Pollution     │
│  (temps réel)             (Open-Meteo)        (Open-Meteo AQ)   │
└────────┬──────────────────────┬───────────────────┬────────────┘
         │                      │                   │
         ▼                      ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                          EXTRACTION                             │
│                                                                 │
│   ecotraffic/load_to_mongodb.py        app/pipeline.py          │
│   (TrafficAPIClient)                   (WeatherClient           │
│                                         + PollutionClient)      │
└────────┬──────────────────────────────────────┬────────────────┘
         │                                      │
         ▼                                      ▼
┌──────────────────┐                 ┌──────────────────────────┐
│    MongoDB       │                 │         MongoDB           │
│  rennes_traffic  │                 │      weather_pollution    │
│  _raw            │                 │      (météo + pollution   │
│  (données brutes)│                 │       fusionnées)         │
└────────┬─────────┘                 └──────────────┬───────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRANSFORMATION (Kedro)                     │
│                                                                 │
│  Pipeline data_processing          Pipeline preprocessing       │
│  - fetch depuis MongoDB            - extract depuis MongoDB     │
│  - parse + clean trafic            - preprocess env features    │
│  - features temporelles            - features engineered        │
│         ↓                                     ↓                 │
│  traffic_cleaned.csv               environment_features         │
│  (data/03_primary/)                (PostgreSQL)                 │
└────────┬──────────────────────────────────────┬────────────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION (Airflow)                     │
│                                                                 │
│  DAG ecotraffic_ingestion_preprocessing  →  @hourly             │
│    load_raw_data_into_mongodb                                   │
│    → run_kedro_preprocessing                                    │
│    → validate_kedro_output                                      │
│    → quality_check                                              │
│    → load_traffic_to_postgres   ← traffic_features (PostgreSQL) │
│                                                                 │
│  DAG ecotraffic_full_pipeline            →  @daily              │
│    extract_to_mongo                                             │
│    → transform_with_kedro                                       │
│    → check_postgres_table                                       │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MACHINE LEARNING                            │
│                                                                 │
│  Modèle Trafic [A FAIRE]          Modèle Environnement [FAIT]   │
│  - target : Statut Trafic         - target : nitrogen_dioxide   │
│  - algo   : Random Forest         - algo   : XGBoost Regression │
│             XGBoost               - R² = 0.758 / MAE = 0.587   │
│         ↓                                     ↓                 │
│    congestion_score               pollution_score (NO2 µg/m³)   │
│         └──────────────┬──────────────────────┘                 │
│                        ▼                                        │
│                 EcoTraffic Score             [A FAIRE]          │
│           (Rouge / Orange / Vert par zone)                      │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DASHBOARD (Streamlit)                       │
│                                                                 │
│  - KPIs : EcoTraffic Score, vitesse moyenne, qualité air        │
│  - Graphiques trafic et environnement                           │
│  - Refresh automatique toutes les heures                        │
│  - Design inspiré Donezo (dark green featured card)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack technique

| Couche | Outil | Rôle |
|---|---|---|
| Langage | Python 3.11 | — |
| Extraction | `requests` | Appels API REST |
| Transformation | Pandas + **Kedro** | Nettoyage, features engineering |
| Stockage brut | **MongoDB** | Collections NoSQL (trafic + météo/pollution) |
| Stockage final | **PostgreSQL** | Tables `environment_features` + `traffic_features` |
| Orchestration | **Apache Airflow** | 2 DAGs (horaire + quotidien) |
| Conteneurisation | **Docker** + `docker-compose.merged.yml` | MongoDB + PostgreSQL + Airflow |
| Tests | **Pytest** | 107 tests unitaires et d'intégration |
| ML | **XGBoost** + scikit-learn + matplotlib | Modèle NO2 entraîné et visualisé |
| Dashboard | **Streamlit** | Visualisation KPIs + graphiques temps réel |
| CI/CD | **GitLab CI** | Build → Test → Deploy |

---

## Structure du projet

```
EcoTraffic/
│
├── app/                              # Pipeline ETL environnement
│   ├── pipeline.py                   # Point d'entrée principal
│   ├── extract/
│   │   ├── weather.py                # Extraction météo (Open-Meteo)
│   │   └── pollution.py              # Extraction pollution (Open-Meteo AQ)
│   ├── transform/
│   │   ├── merge.py                  # Fusion météo + pollution
│   │   └── features.py               # Feature engineering
│   ├── load/
│   │   └── mongo_loader.py           # Chargement MongoDB
│   └── config/
│       └── settings.py               # Configuration
│
├── ecotraffic/                       # Pipeline Kedro — trafic
│   ├── src/ecotraffic/pipelines/
│   │   └── data_processing/
│   │       ├── nodes.py              # Nettoyage et parsing trafic
│   │       └── pipeline.py           # Définition du pipeline Kedro
│   ├── load_to_mongodb.py            # Ingestion trafic → MongoDB
│   └── data/
│       └── 03_primary/
│           └── traffic_cleaned.csv   # Données trafic nettoyées
│
├── kedro_preprocessing/              # Pipeline Kedro — environnement
│   ├── src/ecotraffic_kedro/pipelines/
│   │   └── preprocessing/
│   │       ├── nodes.py              # Nettoyage météo + pollution + save PostgreSQL
│   │       └── pipeline.py           # Définition du pipeline Kedro
│   └── data/
│       ├── 01_raw/raw_env_df.csv
│       └── 02_intermediate/preprocessed_env_df.csv
│
├── airflow/
│   └── dags/
│       ├── ecotraffic_ingestion_preprocessing.py   # DAG trafic (@hourly)
│       │                                           # → load_traffic_to_postgres
│       └── etl_orchestration.py                    # DAG env (@daily)
│
├── ml/                               # Machine Learning
│   ├── env_model.py                  # Modèle XGBoost — prédiction NO2
│   ├── env_model_viz.py              # Visualisation des résultats
│   └── models/
│       ├── env_no2_model.pkl         # Modèle entraîné (sauvegardé)
│       └── env_model_results.png     # Graphiques des résultats
│
├── tests/                            # Tests automatiques (107 tests)
│   ├── test_data_traffic.py          # Qualité données trafic
│   ├── test_data_env.py              # Qualité données météo+pollution
│   ├── test_env_model.py             # Tests du modèle ML (19 tests)
│   ├── test_merge.py                 # Tests fusion météo+pollution
│   ├── test_traffic_nodes.py         # Tests pipeline trafic Kedro
│   ├── test_env_nodes.py             # Tests pipeline environnement Kedro
│   ├── test_postgres_nodes.py        # Tests chargement PostgreSQL + extraction MongoDB
│   ├── test_extract.py               # Tests extraction APIs
│   └── test_load.py                  # Tests chargement MongoDB
│
├── streamlit_app.py                  # Dashboard de visualisation
├── .env                              # Variables d'environnement
├── .gitlab-ci.yml                    # CI/CD : Build → Test → Deploy
├── pytest.ini                        # Configuration pytest
├── docker-compose.merged.yml         # Stack complète (recommandé)
├── requirements.txt
└── README.md
```

---

## Données disponibles

### Trafic (traffic_cleaned.csv → PostgreSQL `traffic_features`)

| Colonne | Description |
|---|---|
| `Date/Heure` | Horodatage de la mesure |
| `Route` | Nom de la route (89 routes uniques) |
| `Vitesse Moyenne (km/h)` | Vitesse moyenne mesurée |
| `Vitesse Max (km/h)` | Vitesse limite de la route |
| `Temps de Trajet (s)` | Durée estimée du trajet |
| `Fiabilité (%)` | Fiabilité de la mesure |
| `Statut Trafic` | freeflow / heavy / congested / unknown |
| `Hiérarchie` | Type de réseau routier |
| `datetime_hour` | Heure arrondie (clé de jointure) |

### Environnement (PostgreSQL `environment_features`)

| Colonne | Description |
|---|---|
| `temperature_2m` | Température à 2m en °C |
| `precipitation` | Précipitations en mm |
| `wind_speed_10m` / `wind_kmh` | Vitesse du vent |
| `pm10` / `pm2_5` | Particules en suspension (µg/m³) |
| `carbon_monoxide` | CO en µg/m³ |
| `nitrogen_dioxide` | NO2 en µg/m³ |
| `ozone` | O3 en µg/m³ |
| `hour` / `day_of_week` / `is_weekend` | Features temporelles |
| `rain_flag` | 1 si précipitations > 0 |

---

## Machine Learning — Prédiction NO₂

### Objectif

Prédire le taux de **dioxyde d'azote (NO₂)** à Rennes en µg/m³ à partir des conditions météo et de la pollution ambiante.

### Modèle

| Paramètre | Valeur |
|---|---|
| Algorithme | XGBoost Regression |
| Target | `nitrogen_dioxide` (µg/m³) |
| Dataset | 120 lignes — mai 2026 |
| Split | 80% train / 20% test |

### Justification du choix — XGBoost

| Raison | Explication |
|---|---|
| **Petit dataset** | 120 lignes seulement. XGBoost intègre une régularisation L1/L2 qui évite l'overfitting. |
| **Features hétérogènes** | Mélange continu, entiers et binaires — aucune normalisation requise. |
| **Relations non-linéaires** | Pics NO2 aux heures de pointe non capturables par une régression linéaire. |
| **Interprétabilité** | Importance des features native (CO = 55%, ozone = 34%). |
| **Rapidité** | Entraînement en < 1 seconde, ré-entraînable à chaque collecte. |

### Résultats

| Métrique | Valeur | Interprétation |
|---|---|---|
| **R²** | **0.758** | Le modèle explique 75.8% de la variance du NO2 |
| **MAE** | 0.587 µg/m³ | Erreur moyenne de 0.6 µg/m³ |
| **RMSE** | 0.886 µg/m³ | Erreur quadratique |

---

## CI/CD — GitLab

Le pipeline CI/CD suit l'ordre **Build → Test → Deploy** :

```
build  →  construit l'image Docker (airflow/Dockerfile)
  ↓         et sauvegarde l'artifact (image.tar.gz)
test   →  installe les dépendances Python et lance pytest
  ↓         107 tests doivent passer pour continuer
deploy →  déploiement manuel sur EC2 (uniquement sur main)
```

| Stage | Déclencheur | Image |
|---|---|---|
| `build` | Chaque push | `docker:latest` |
| `test` | Après build réussi | `python:3.11` |
| `deploy` | Manuel, branche `main` uniquement | `alpine:latest` |

---

## Avancement du projet

| Composant | Statut |
|---|---|
| Extraction trafic (API Rennes) | Fait |
| Extraction météo (Open-Meteo) | Fait |
| Extraction pollution (Open-Meteo AQ) | Fait |
| Transformation Kedro — trafic | Fait |
| Transformation Kedro — environnement | Fait |
| Fusion météo + pollution | Fait |
| Stockage MongoDB (trafic + environnement) | Fait |
| Stockage PostgreSQL `environment_features` | Fait |
| Stockage PostgreSQL `traffic_features` | Fait |
| DAG Airflow trafic (@hourly) + injection PostgreSQL | Fait |
| DAG Airflow environnement (@daily) | Fait |
| Tests automatiques (107 tests) | Fait |
| Conteneurisation Docker | Fait |
| CI/CD GitLab (Build → Test → Deploy) | Fait |
| **Modèle ML environnement (NO2)** | **Fait** |
| **Dashboard Streamlit** | **Fait** |
| Fusion trafic + environnement | En cours |
| Modèle ML trafic | A faire |
| EcoTraffic Score combiné | A faire |

---

## Lancer le projet

### Avec Docker et Airflow (recommandé)

```bash
docker compose -f docker-compose.merged.yml up --build
```

Interface Airflow : `http://localhost:8080`  
Identifiants : `admin` / `admin123`

### Dashboard Streamlit

```bash
pip install streamlit
streamlit run streamlit_app.py
```

### En local (sans Docker)

```bash
# Pipeline environnement (météo + pollution)
python -m app.pipeline

# Pipeline trafic (ingestion + Kedro)
cd ecotraffic
python load_to_mongodb.py
kedro run --pipeline data_processing
```

### Modèle ML — Prédiction NO2

```bash
python ml/env_model.py
python ml/env_model_viz.py
```

---

## Variables d'environnement (.env)

```env
# MongoDB
MONGO_URI=mongodb://mongo:27017/ecotraffic
MONGO_DB=ecotraffic
MONGO_COLLECTION=traffic
MONGO_COLLECTION_ENV=weather_pollution

# PostgreSQL
POSTGRES_URI=postgresql+psycopg2://postgres:postgres@postgres:5432/ecotraffic
POSTGRES_TABLE_ENV=environment_features
POSTGRES_TABLE_TRAFFIC=traffic_features

# APIs
WEATHER_API_URL=https://api.open-meteo.com/v1/forecast
AIR_QUALITY_API_URL=https://air-quality-api.open-meteo.com/v1/air-quality
API_BASE_URL=https://data.rennesmetropole.fr/api/records/1.0/search/

# Localisation Rennes
LATITUDE=48.1173
LONGITUDE=-1.6778
```

---

## Tests

### Lancer les tests

```bash
# Tous les tests
pytest tests/ -v --tb=short

# Un fichier précis
pytest tests/test_postgres_nodes.py -v

# Arrêter au premier échec
pytest tests/ -v -x
```

Résultat attendu : **107 passed**

---

### Couverture des tests

| Fichier | Tests | Ce qui est testé |
|---|---|---|
| `test_data_traffic.py` | 17 | Qualité des données trafic (colonnes, plages, statuts, cohérence temporelle) |
| `test_data_env.py` | 22 | Qualité des données météo + pollution (nulls, doublons, plages, flags) |
| `test_env_model.py` | 19 | Modèle ML : chargement, entraînement, métriques, prédiction, sauvegarde |
| `test_merge.py` | 13 | Fusion météo + pollution, flags, cas d'erreur |
| `test_traffic_nodes.py` | 14 | Parsing et nettoyage des données trafic Kedro |
| `test_env_nodes.py` | 11 | Preprocessing des données environnement Kedro |
| `test_postgres_nodes.py` | 8 | Chargement PostgreSQL (trafic + env) + extraction MongoDB (mocks) |
| `test_extract.py` | 2 | Extraction météo et pollution (mocks HTTP) |
| `test_load.py` | 1 | Chargement MongoDB |

---

## Auteurs

**Eya Ben Salem** — Master Big Data & IA  
**Yasmine Amor** — Master Big Data & IA
