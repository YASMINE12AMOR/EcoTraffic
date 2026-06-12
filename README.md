<div align="center">

# 🌍 EcoTraffic

### Pipeline de données & Machine Learning — Rennes

*Collecte en temps réel des données de **trafic routier**, **météo** et **qualité de l'air** sur la ville de Rennes, avec un pipeline ETL automatisé et un modèle ML de prédiction de la qualité de l'air.*

<br>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Kedro](https://img.shields.io/badge/Kedro-FFC900?style=flat-square&logo=kedro&logoColor=black)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Airflow](https://img.shields.io/badge/Apache_Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-EB0F00?style=flat-square&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-107_tests-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![GitLab CI](https://img.shields.io/badge/GitLab_CI-FC6D26?style=flat-square&logo=gitlab&logoColor=white)

![Model R²](https://img.shields.io/badge/Modèle_NO₂_R²-0.758-success?style=flat-square)
![Routes](https://img.shields.io/badge/Routes_Rennes-89-blue?style=flat-square)
![Localisation](https://img.shields.io/badge/Rennes-48.1173,_--1.6778-informational?style=flat-square)

</div>

---

## 🎯 Objectifs

- 📡 Collecter en temps réel les données trafic, météo et pollution via des APIs
- 🧹 Nettoyer, transformer et stocker les données dans des bases structurées
- ⚙️ Automatiser l'exécution du pipeline avec Apache Airflow
- 🤖 Appliquer un modèle ML pour prédire la qualité de l'air (NO₂)
- 🌱 Poser les bases d'un **EcoTraffic Score** combinant trafic et environnement *(en cours)*

---

## 📡 Sources de données

| API | Données | Fréquence | Localisation |
|---|---|---|---|
| [Rennes Métropole OpenData](https://data.rennesmetropole.fr/api/records/1.0/search/) | Trafic temps réel (vitesse, statut, routes) | Temps réel | 89 routes de Rennes |
| [Open-Meteo Weather](https://api.open-meteo.com/v1/forecast) | Météo horaire (température, pluie, vent) | Horaire | Rennes `48.1173, -1.6778` |
| [Open-Meteo Air Quality](https://air-quality-api.open-meteo.com/v1/air-quality) | Pollution horaire (PM2.5, PM10, NO₂, CO, O₃) | Horaire | Rennes `48.1173, -1.6778` |

<details>
<summary><b>🔍 Détail des APIs</b></summary>

<br>

**🚗 Trafic — Rennes Métropole OpenData**
- **URL** : `https://data.rennesmetropole.fr/api/records/1.0/search/`
- **Accès** : Public, sans authentification
- **Données** : vitesse moyenne, vitesse max, temps de trajet, statut trafic (freeflow / heavy / congested), hiérarchie de la route
- **Couverture** : 89 routes du réseau de Rennes Métropole

**🌤️ Météo — Open-Meteo Weather**
- **URL** : `https://api.open-meteo.com/v1/forecast`
- **Accès** : Public, sans authentification
- **Paramètres collectés** : `temperature_2m`, `precipitation`, `wind_speed_10m`
- **Résolution** : horaire, coordonnées GPS `(48.1173, -1.6778)`

**🏭 Qualité de l'air — Open-Meteo Air Quality**
- **URL** : `https://air-quality-api.open-meteo.com/v1/air-quality`
- **Accès** : Public, sans authentification
- **Paramètres collectés** : `pm10`, `pm2_5`, `carbon_monoxide`, `nitrogen_dioxide`, `ozone`, `sulphur_dioxide`
- **Résolution** : horaire, coordonnées GPS `(48.1173, -1.6778)`

</details>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         SOURCES DE DONNÉES                       │
│                                                                  │
│  API Trafic Rennes        API Météo           API Pollution      │
│  (temps réel)             (Open-Meteo)        (Open-Meteo AQ)     │
└────────┬──────────────────────┬───────────────────┬─────────────┘
         │                      │                    │
         ▼                      ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                          EXTRACTION                              │
│                                                                  │
│   ecotraffic/load_to_mongodb.py        app/pipeline.py           │
│   (TrafficAPIClient)                   (WeatherClient            │
│                                         + PollutionClient)       │
└────────┬──────────────────────────────────────┬─────────────────┘
         │                                       │
         ▼                                       ▼
┌──────────────────┐                 ┌──────────────────────────┐
│    MongoDB       │                 │         MongoDB          │
│  rennes_traffic  │                 │      weather_pollution   │
│  _raw            │                 │      (météo + pollution  │
│  (données brutes)│                 │       fusionnées)        │
└────────┬─────────┘                 └──────────────┬───────────┘
         │                                          │
         ▼                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TRANSFORMATION (Kedro)                      │
│                                                                  │
│  Pipeline data_processing          Pipeline preprocessing        │
│  - fetch depuis MongoDB            - extract depuis MongoDB       │
│  - parse + clean trafic            - preprocess env features      │
│  - features temporelles            - features engineered          │
│         ↓                                     ↓                  │
│  traffic_cleaned.csv               environment_features          │
│  (data/03_primary/)                (PostgreSQL)                  │
└────────┬──────────────────────────────────────┬─────────────────┘
         │                                       │
         └──────────────┬────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATION (Airflow)                      │
│                                                                  │
│  DAG ecotraffic_ingestion_preprocessing  →  @hourly              │
│    load_raw_data_into_mongodb                                    │
│    → run_kedro_preprocessing                                     │
│    → validate_kedro_output                                       │
│    → quality_check                                               │
│    → load_traffic_to_postgres   ← traffic_features (PostgreSQL)  │
│                                                                  │
│  DAG ecotraffic_full_pipeline            →  @daily               │
│    extract_to_mongo                                              │
│    → transform_with_kedro                                        │
│    → check_postgres_table                                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MACHINE LEARNING                             │
│                                                                  │
│  Modèle Environnement                                            │
│  - target : nitrogen_dioxide (NO₂)                               │
│  - algo   : XGBoost Regression                                   │
│  - R² = 0.758 / MAE = 0.587 µg/m³                                │
│         ↓                                                        │
│  pollution_score (NO₂ µg/m³)                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DASHBOARD (Streamlit)                        │
│                                                                  │
│  - KPIs : EcoTraffic Score, vitesse moyenne, qualité air         │
│  - Graphiques trafic et environnement                            │
│  - Refresh automatique toutes les heures                         │
│  - Design inspiré Donezo (dark green featured card)              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack technique

| Couche | Outil | Rôle |
|---|---|---|
| 🐍 Langage | Python 3.11 | — |
| 📥 Extraction | `requests` | Appels API REST |
| 🔄 Transformation | Pandas + **Kedro** | Nettoyage, feature engineering |
| 🍃 Stockage brut | **MongoDB** | Collections NoSQL (trafic + météo/pollution) |
| 🐘 Stockage final | **PostgreSQL** | Tables `environment_features` + `traffic_features` |
| ⚙️ Orchestration | **Apache Airflow** | 2 DAGs (horaire + quotidien) |
| 🐳 Conteneurisation | **Docker** + `docker-compose.merged.yml` | MongoDB + PostgreSQL + Airflow |
| ✅ Tests | **Pytest** | 107 tests unitaires et d'intégration |
| 🤖 ML | **XGBoost** + scikit-learn + matplotlib | Modèle NO₂ entraîné et visualisé |
| 📊 Dashboard | **Streamlit** | Visualisation KPIs + graphiques temps réel |
| 🚀 CI/CD | **GitLab CI** | Build → Test → Deploy |

---

## 📁 Structure du projet

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
│   ├── env_model.py                  # Modèle XGBoost — prédiction NO₂
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
├── monitoring/
│   └── prometheus.yml                # Config Prometheus + cAdvisor (monitoring Docker)
│
├── scripts/
│   └── build_traffic_cleaned.py      # Bypass API → CSV direct (sans MongoDB)
│
├── streamlit_app.py                  # Dashboard de visualisation
├── .env                              # Variables d'environnement (à créer manuellement)
├── .gitlab-ci.yml                    # CI/CD : Build → Test → Deploy
├── pytest.ini                        # Configuration pytest
├── docker-compose.merged.yml         # Stack complète recommandée (LocalExecutor)
├── docker-compose.airflow.yml        # Ancienne stack (déprécié — SequentialExecutor)
├── requirements.txt                  # Dépendances Python principales
├── requirements-airflow.txt          # Dépendances Airflow
├── requirements-dashboard.txt        # Dépendances Streamlit
└── README.md
```

---

## 📊 Données disponibles

### 🚗 Trafic — `traffic_cleaned.csv` → PostgreSQL `traffic_features`

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

### 🌤️ Environnement — PostgreSQL `environment_features`

| Colonne | Description |
|---|---|
| `temperature_2m` | Température à 2m en °C |
| `precipitation` | Précipitations en mm |
| `wind_speed_10m` / `wind_kmh` | Vitesse du vent |
| `pm10` / `pm2_5` | Particules en suspension (µg/m³) |
| `carbon_monoxide` | CO en µg/m³ |
| `nitrogen_dioxide` | NO₂ en µg/m³ |
| `ozone` | O₃ en µg/m³ |
| `hour` / `day_of_week` / `is_weekend` | Features temporelles |
| `rain_flag` | 1 si précipitations > 0 |

---

## 🤖 Machine Learning — Prédiction NO₂

> **Objectif** — Prédire le taux de **dioxyde d'azote (NO₂)** à Rennes en µg/m³ à partir des conditions météo et de la pollution ambiante.

### ⚙️ Modèle

| Paramètre | Valeur |
|---|---|
| Algorithme | XGBoost Regression |
| Target | `nitrogen_dioxide` (µg/m³) |
| Dataset | 120 lignes — mai 2026 |
| Split | 80% train / 20% test |

<details>
<summary><b>💡 Pourquoi XGBoost ?</b></summary>

<br>

| Raison | Explication |
|---|---|
| **Petit dataset** | 120 lignes seulement. XGBoost intègre une régularisation L1/L2 qui évite l'overfitting. |
| **Features hétérogènes** | Mélange continu, entiers et binaires — aucune normalisation requise. |
| **Relations non-linéaires** | Pics NO₂ aux heures de pointe non capturables par une régression linéaire. |
| **Interprétabilité** | Importance des features native (CO = 55%, ozone = 34%). |
| **Rapidité** | Entraînement en < 1 seconde, ré-entraînable à chaque collecte. |

</details>

### 📈 Résultats

| Métrique | Valeur | Interprétation |
|:---:|:---:|---|
| **R²** | **0.758** | Le modèle explique 75.8% de la variance du NO₂ |
| **MAE** | **0.587 µg/m³** | Erreur moyenne de 0.6 µg/m³ |
| **RMSE** | **0.886 µg/m³** | Erreur quadratique |

---

## 🚀 CI/CD — GitLab

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

## ▶️ Lancer le projet

### 🐳 Avec Docker et Airflow *(recommandé)*

```bash
docker compose -f docker-compose.merged.yml up --build
```

> Interface Airflow : `http://localhost:8080` — Identifiants : `admin` / `admin123`

### 📊 Dashboard Streamlit

```bash
pip install streamlit
streamlit run streamlit_app.py
```

### 💻 En local *(sans Docker)*

```bash
# Pipeline environnement (météo + pollution)
python -m app.pipeline

# Pipeline trafic (ingestion + Kedro)
cd ecotraffic
python load_to_mongodb.py
kedro run --pipeline data_processing
```

### 🤖 Modèle ML — Prédiction NO₂

```bash
python ml/env_model.py
python ml/env_model_viz.py
```

---

## ✅ Tests

```bash
# Tous les tests
pytest tests/ -v --tb=short

# Un fichier précis
pytest tests/test_postgres_nodes.py -v

# Arrêter au premier échec
pytest tests/ -v -x
```

<details>
<summary><b>📋 Couverture des tests (107 tests)</b></summary>

<br>

### 📊 Résumé des tests par module

| Fichier | # Tests | Catégorie | Description |
|---|:---:|---|---|
| `test_basic.py` | 1 | Fondamental | Test trivial d'assertion |
| `test_extract.py` | 2 | ETL | Extraction météo et pollution (mocks HTTP) |
| `test_load.py` | 1 | ETL | Chargement MongoDB (mongomock) |
| `test_merge.py` | 13 | Transformation | Fusion météo + pollution, création features, validation cas d'erreur |
| `test_data_env.py` | 22 | Qualité | Données environnement : colonnes, nulls, doublons, plages (T°, pluie, vent, polluants), cohérence flags |
| `test_data_traffic.py` | 17 | Qualité | Données trafic : colonnes, plages vitesse (0-130 km/h), statuts valides, pas de doublons |
| `test_env_model.py` | 9 | ML | Modèle XGBoost : chargement données, entraînement, split 80/20, métriques (MAE < 2, R² > 0.5), cross-validation |
| `test_env_nodes.py` | 15 | Kedro (Env) | Preprocessing environnement : suppression colonnes, ajout features (hour, day_of_week, month, is_weekend, rain_flag), tri, déduplication |
| `test_traffic_nodes.py` | 14 | Kedro (Trafic) | Parsing MongoDB (flat + nested), nettoyage statuts (HEAVY → heavy), extraction champs avec fallback, ajout datetime_hour |
| `test_postgres_nodes.py` | 13 | Integration | Chargement PostgreSQL (env + trafic), extraction MongoDB, gestion DataFrames vides (mocks) |
| **TOTAL** | **107** | — | — |

### 🎯 Couverture fonctionnelle

| Flux | Tests | Couverture |
|---|:---:|---|
| **Extraction (APIs)** | 2 | ✅ Météo + Pollution |
| **Chargement (MongoDB)** | 1 | ✅ MongoDB insertion |
| **Transformation (Merge)** | 13 | ✅ Fusion, features, erreurs |
| **Qualité données (Env)** | 22 | ✅ 100% colonnes, nulls, plages, cohérence |
| **Qualité données (Trafic)** | 17 | ✅ 100% colonnes, vitesse, statuts |
| **Preprocessing Kedro (Env)** | 15 | ✅ Features temporelles, flags, déduplication |
| **Preprocessing Kedro (Trafic)** | 14 | ✅ Parsing, normalisation, extraction |
| **ML (XGBoost NO₂)** | 9 | ✅ Entraînement, métriques, prédictions |
| **PostgreSQL + MongoDB** | 13 | ✅ Chargement, extraction, mocks |

</details>

---

## 👥 Auteurs

| | |
|---|---|
| **Eya Ben Salem** | Master Big Data & IA |
| **Yasmine Amor** | Master Big Data & IA |

<div align="center">

<br>

*🌍 EcoTraffic — Données, environnement et mobilité au service de Rennes*

</div>