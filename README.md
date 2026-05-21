# EcoTraffic — Pipeline de données & Machine Learning (Rennes)

Projet de Data Engineering et Machine Learning collectant des données de **trafic routier**, de **météo** et de **qualité de l'air** sur la ville de **Rennes**, avec un pipeline ETL automatisé et une couche ML prédictive en cours de développement.

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
│  Modèle Trafic                    Modèle Environnement          │
│  - target : Statut Trafic         - target : nitrogen_dioxide   │
│    ou Vitesse Moyenne               ou pm2_5                    │
│  - algo   : Random Forest         - algo   : XGBoost / LSTM     │
│             XGBoost                                             │
│         ↓                                     ↓                 │
│    congestion_score               pollution_score               │
│         └──────────────┬──────────────────────┘                 │
│                        ▼                                        │
│                 EcoTraffic Score                                │
│           (Rouge / Orange / Vert par zone)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack technique

| Couche | Outil | Rôle |
|---|---|---|
| Langage | Python 3 | — |
| Extraction | `requests` | Appels API REST |
| Transformation | Pandas + **Kedro** | Nettoyage, features engineering |
| Stockage brut | **MongoDB** | Collections NoSQL (trafic + météo) |
| Stockage final | **PostgreSQL** | Table `environment_features` |
| Orchestration | **Apache Airflow** | 2 DAGs (horaire + quotidien) |
| Conteneurisation | **Docker** | `docker-compose.yml` |
| Tests | Pytest | Tests unitaires et d'intégration |
| ML (en cours) | scikit-learn / XGBoost | Modèles prédictifs |

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
│   │       ├── nodes.py              # Nettoyage météo + pollution
│   │       └── pipeline.py           # Définition du pipeline Kedro
│   └── data/
│       ├── 01_raw/raw_env_df.csv
│       └── 02_intermediate/preprocessed_env_df.csv
│
├── airflow/
│   └── dags/
│       ├── ecotraffic_ingestion_preprocessing.py   # DAG trafic (@hourly)
│       └── etl_orchestration.py                    # DAG env (@daily)
│
├── .env                              # Variables d'environnement
├── docker-compose.yml                # MongoDB + PostgreSQL + Airflow
├── requirements.txt
└── README.md
```

---

## Données disponibles

### Trafic (traffic_cleaned.csv)

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

### Environnement (preprocessed_env_df.csv)

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

## Lancer le projet

### En local

```bash
# Pipeline environnement (météo + pollution)
python -m app.pipeline

# Pipeline trafic (ingestion + Kedro)
cd ecotraffic
python load_to_mongodb.py
kedro run --pipeline data_processing
```

### Avec Docker et Airflow

```bash
docker compose up --build
```

Interface Airflow : `http://localhost:8080`  
Identifiants : `admin` / `admin123`

---

## Variables d'environnement (.env)

```env
# MongoDB
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/ecotraffic
MONGO_DB=ecotraffic
MONGO_COLLECTION=weather_pollution

# PostgreSQL
POSTGRES_URI=postgresql+psycopg2://postgres:postgres@postgres:5432/ecotraffic

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
python -m pytest tests/ -v

# Uniquement les tests sur les données réelles
python -m pytest tests/test_data_traffic.py tests/test_data_env.py -v

# Arrêter au premier échec
python -m pytest tests/ -v -x
```

Résultat attendu : **80 passed**

---

### Couverture des tests

| Fichier | Tests | Ce qui est testé |
|---|---|---|
| `tests/test_data_traffic.py` | 17 | Qualité des données réelles trafic (traffic_cleaned.csv) |
| `tests/test_data_env.py` | 22 | Qualité des données réelles météo + pollution (preprocessed_env_df.csv) |
| `tests/test_merge.py` | 13 | Fusion météo + pollution, flags, erreurs |
| `tests/test_traffic_nodes.py` | 14 | Parsing et nettoyage des données trafic |
| `tests/test_env_nodes.py` | 11 | Preprocessing des données environnement |
| `tests/test_extract.py` | 2 | Extraction météo et pollution (mocks HTTP) |
| `tests/test_load.py` | 1 | Chargement MongoDB |

### Tests sur les données réelles

**Trafic (`test_data_traffic.py`) :**
- Fichier chargeable, 356 lignes, 10 colonnes présentes
- Pas de valeurs nulles sur Date/Heure, Route, Vitesse
- Pas de doublons
- Vitesse Moyenne entre 0 et 130 km/h
- Statuts valides (`freeflow`, `heavy`, `congested`, `unknown`) et en minuscules
- Hiérarchies connues, routes non vides
- `datetime_hour` cohérent avec `Date/Heure`

**Météo + Pollution (`test_data_env.py`) :**
- Fichier chargeable, 120 lignes, 18 colonnes présentes
- Pas de nulls ni doublons sur `datetime`
- Données triées chronologiquement
- Température entre -10°C et 50°C
- PM2.5, PM10, NO2, ozone tous ≥ 0
- `wind_kmh = wind_speed_10m × 3.6`
- `high_pm2_5` cohérent avec seuil OMS (25 µg/m³)
- `high_pm10` cohérent avec seuil OMS (50 µg/m³)
- `rain_flag` et `is_weekend` binaires
- `month` cohérent avec `datetime`

---

## Auteurs

**Eya Ben Salem** — Master Big Data & IA  
**Yasmine Amor** — Master Big Data & IA
