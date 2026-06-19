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
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square&logo=lightgbm&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-130_tests-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![GitLab CI](https://img.shields.io/badge/GitLab_CI-FC6D26?style=flat-square&logo=gitlab&logoColor=white)

![Model R²](https://img.shields.io/badge/Modèle_NO₂_R²-0.892-success?style=flat-square)
![Models Compared](https://img.shields.io/badge/Modèles_comparés-6-blue?style=flat-square)
![Routes](https://img.shields.io/badge/Routes_Rennes-89-blue?style=flat-square)
![Localisation](https://img.shields.io/badge/Rennes-48.1173,_--1.6778-informational?style=flat-square)

</div>

---

## 🎯 Objectifs

- 📡 Collecter en temps réel les données trafic, météo et pollution via des APIs
- 🧹 Nettoyer, transformer et stocker les données dans des bases structurées
- ⚙️ Automatiser l'exécution du pipeline avec Apache Airflow
- 🤖 Comparer 6 modèles ML et sélectionner le meilleur pour prédire la qualité de l'air (NO₂)
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
│    → compare_ml_models          ← models_comparison.json         │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MACHINE LEARNING                             │
│                                                                  │
│  Comparaison de 6 modèles (compare_models.py)                    │
│  XGBoost · LightGBM · CatBoost · Random Forest · Ridge · MLP    │
│         ↓                                                        │
│  Meilleur modèle : XGBoost Regression                            │
│  - target : nitrogen_dioxide (NO₂)                               │
│  - R² = 0.892 / MAE = 0.655 µg/m³                                │
│         ↓                                                        │
│  pollution_score (NO₂ µg/m³)                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     DASHBOARD (Streamlit)                        │
│                                                                  │
│  - Conteneurisé (Docker) — accessible sur localhost:8501          │
│  - 7 onglets : Vue globale, Trafic, Carte, Modele ML,           │
│    Prediction NO2, Actions anti-pollution, Qualite des donnees   │
│  - Comparaison interactive de 6 modèles ML (graphiques Plotly)   │
│  - Visualisation dynamique du modèle XGBoost (auto-générée)      │
│  - Actions anti-pollution adaptées au niveau de risque           │
│  - Refresh automatique toutes les heures                         │
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
| 🐳 Conteneurisation | **Docker** + `docker-compose.merged.yml` | 7 services (MongoDB, PostgreSQL, Airflow, Streamlit, Prometheus, Grafana, cAdvisor) |
| ✅ Tests | **Pytest** | 130 tests unitaires et d'intégration |
| 🤖 ML | **XGBoost** + LightGBM + CatBoost + scikit-learn | 6 modèles comparés, XGBoost retenu pour NO₂ |
| 📊 Dashboard | **Streamlit** (conteneurisé) | 7 onglets : vue globale, trafic, carte, ML, prédiction, actions, qualité |
| 📈 Monitoring | **Prometheus** + **Grafana** + **cAdvisor** | Monitoring Docker (CPU, mémoire, réseau) |
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
│   ├── compare_models.py             # Comparaison de 6 modèles ML
│   ├── env_model_viz.py              # Visualisation des résultats (matplotlib)
│   ├── monitor_drift.py              # Détection de data drift
│   └── models/
│       ├── env_no2_model.pkl         # Modèle entraîné (sauvegardé)
│       ├── models_comparison.json    # Résultats comparaison des 6 modèles
│       ├── reference_stats.json      # Statistiques de référence (drift)
│       └── drift_report.json         # Rapport de data drift
│
├── tests/                            # Tests automatiques (130 tests)
│   ├── test_compare_models.py        # Tests comparaison 6 modèles ML (21 tests)
│   ├── test_data_traffic.py          # Qualité données trafic
│   ├── test_data_env.py              # Qualité données météo+pollution
│   ├── test_env_model.py             # Tests du modèle ML (19 tests)
│   ├── test_merge.py                 # Tests fusion météo+pollution
│   ├── test_traffic_nodes.py         # Tests pipeline trafic Kedro
│   ├── test_env_nodes.py             # Tests pipeline environnement Kedro
│   ├── test_postgres_nodes.py        # Tests chargement PostgreSQL + extraction MongoDB
│   ├── test_extract.py               # Tests extraction APIs
│   └── test_load.py                  # Tests chargement MongoDB (upsert, doublons)
│
├── monitoring/
│   └── prometheus.yml                # Config Prometheus + cAdvisor (monitoring Docker)
│
├── scripts/
│   └── build_traffic_cleaned.py      # Bypass API → CSV direct (sans MongoDB)
│
├── streamlit_app.py                  # Dashboard de visualisation (7 onglets)
├── Dockerfile.streamlit              # Image Docker pour Streamlit
├── .env                              # Variables d'environnement (à créer manuellement)
├── .gitlab-ci.yml                    # CI/CD : Build → Test → Deploy
├── pytest.ini                        # Configuration pytest
├── docker-compose.merged.yml         # Stack complète (7 services conteneurisés)
├── docker-compose.airflow.yml        # Ancienne stack (déprécié)
├── requirements.txt                  # Dépendances Python
├── requirements-airflow.txt          # Dépendances Airflow (dans Docker)
├── requirements-dashboard.txt        # Dépendances Streamlit (dans Docker)
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

### 🔬 Comparaison de 6 modèles

Six algorithmes ont été testés et comparés sur le même jeu de données (829 échantillons depuis PostgreSQL, split 80/20, cross-validation 5-fold) :

| Rang | Modèle | R² | MAE | RMSE | Justification |
|:---:|---|:---:|:---:|:---:|---|
| 🥇 | **XGBoost** | **0.892** | **0.655** | **1.057** | Gradient boosting classique, gère bien les relations non-linéaires |
| 🥈 | CatBoost | 0.871 | 0.695 | — | Conçu pour données hétérogènes, pas besoin d'encodage |
| 🥉 | LightGBM | 0.844 | 0.743 | — | Plus rapide que XGBoost, moins d'overfitting |
| 4 | Neural Network (MLP) | 0.835 | 0.792 | — | Performant avec suffisamment de données |
| 5 | Random Forest | 0.734 | 0.989 | — | Ensemble simple, baseline de comparaison |
| 6 | Ridge Regression | 0.718 | 1.035 | — | Modèle linéaire, sous-performant sur relations non-linéaires |

### ⚙️ Modèle retenu : XGBoost

| Paramètre | Valeur |
|---|---|
| Algorithme | XGBoost Regression |
| Target | `nitrogen_dioxide` (µg/m³) |
| Source données | PostgreSQL (`environment_features`) avec fallback CSV |
| Dataset | 829 échantillons (accumulation historique) |
| Split | 80% train / 20% test |

<details open>
<summary><b>💡 Pourquoi XGBoost ?</b></summary>

<br>

| Raison | Explication |
|---|---|
| **Meilleur R² (0.892)** | Classé 1er parmi les 6 modèles testés. |
| **Dataset croissant** | 829+ échantillons (accumulation automatique). XGBoost intègre une régularisation L1/L2 qui évite l'overfitting. |
| **Features hétérogènes** | Mélange continu, entiers et binaires — aucune normalisation requise. |
| **Relations non-linéaires** | Pics NO₂ aux heures de pointe non capturables par une régression linéaire. |
| **Interprétabilité** | Importance des features native (ozone = 41.7%, hour = 27.5%). |
| **Rapidité** | Entraînement en ~3s, ré-entraînable à chaque collecte. |

</details open>

### 📈 Résultats du meilleur modèle

| Métrique | Valeur | Interprétation |
|:---:|:---:|---|
| **R²** | **0.892** | Le modèle explique 89.2% de la variance du NO₂ |
| **MAE** | **0.655 µg/m³** | Erreur moyenne de 0.65 µg/m³ |
| **RMSE** | **1.057 µg/m³** | Erreur quadratique |

Les visualisations (réel vs prédit, erreurs, importance des features, scénarios par heure) sont générées **automatiquement** dans le dashboard Streamlit à chaque ouverture, sans intervention manuelle.

---

## 🚀 CI/CD — GitLab

Le pipeline CI/CD suit l'ordre **Build → Test → Deploy** :

```
build  →  construit l'image Docker (airflow/Dockerfile)
  ↓         et sauvegarde l'artifact (image.tar.gz)
test   →  installe les dépendances Python et lance pytest
  ↓         130 tests doivent passer pour continuer
deploy →  déploiement manuel sur EC2 (uniquement sur main)
```

<img width="1071" height="160" alt="image" src="https://github.com/user-attachments/assets/0a1fcf7b-3b0a-4366-9808-f1b4d7eff75e" />


| Stage | Déclencheur | Image |
|---|---|---|
| `build` | Chaque push | `docker:latest` |
| `test` | Après build réussi | `python:3.11` |
| `deploy` | Manuel, branche `main` uniquement | `alpine:latest` |

---

## ▶️ Lancer le projet

### Prérequis

- **Docker Desktop** (inclut Docker Compose)
- **Git** (pour cloner le projet)

### 1. Cloner et lancer (une seule commande)

```bash
git clone <url-du-repo>
cd EcoTraffic
docker compose -f docker-compose.merged.yml up --build -d
```

Cela démarre **7 services** automatiquement :

| Service | URL | Identifiants |
|---|---|---|
| **Streamlit** (dashboard) | `http://localhost:8501` | — |
| **Airflow** (orchestration) | `http://localhost:8080` | admin / admin123 |
| **Grafana** (monitoring) | `http://localhost:3000` | admin / admin |
| Prometheus | `http://localhost:9090` | — |
| cAdvisor | `http://localhost:8081` | — |
| MongoDB | `localhost:27017` | — |
| PostgreSQL | `localhost:5432` | postgres / postgres |

> Aucune installation Python locale requise — tout est conteneurisé.

### 2. Lancer le pipeline de données

Dans Airflow (`http://localhost:8080`), activez le DAG **`ecotraffic_full_pipeline`**. Il exécute automatiquement :
1. Collecte météo + pollution (30 jours) → MongoDB
2. Preprocessing Kedro → PostgreSQL
3. Comparaison de 6 modèles ML → `models_comparison.json`

Le dashboard Streamlit se met à jour automatiquement (cache 1h).

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
# Comparaison des 6 modèles (résultats dans ml/models/models_comparison.json)
python ml/compare_models.py

# Entraînement du modèle XGBoost seul
python ml/env_model.py
```

> Les graphiques ML sont générés automatiquement dans le dashboard Streamlit — plus besoin de lancer `env_model_viz.py`.

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

<details open>
<summary><b>📋 Couverture des tests (130 tests)</b></summary>

<br>

### 📊 Résumé des tests par module

| Fichier | # Tests | Catégorie | Description |
|---|:---:|---|---|
| `test_basic.py` | 1 | Fondamental | Test trivial d'assertion |
| `test_extract.py` | 2 | ETL | Extraction météo et pollution (mocks HTTP) |
| `test_load.py` | 3 | ETL | Chargement MongoDB : upsert, pas de doublons, mise à jour existants |
| `test_merge.py` | 13 | Transformation | Fusion météo + pollution, création features, validation cas d'erreur |
| `test_data_env.py` | 22 | Qualité | Données environnement : colonnes, nulls, doublons, plages, cohérence flags |
| `test_data_traffic.py` | 17 | Qualité | Données trafic : colonnes, plages vitesse (0-130 km/h), statuts valides |
| `test_env_model.py` | 19 | ML | Modèle XGBoost : chargement PostgreSQL, entraînement, métriques, prédictions |
| `test_compare_models.py` | 21 | ML | Comparaison 6 modèles : évaluation, cross-validation, JSON valide, classement |
| `test_env_nodes.py` | 11 | Kedro (Env) | Preprocessing environnement : features temporelles, flags, déduplication |
| `test_traffic_nodes.py` | 13 | Kedro (Trafic) | Parsing MongoDB, nettoyage statuts, extraction champs, ajout datetime_hour |
| `test_postgres_nodes.py` | 8 | Integration | Chargement PostgreSQL (append), extraction MongoDB, gestion vides |
| **TOTAL** | **130** | — | — |

### 🎯 Couverture fonctionnelle

| Flux | Tests | Couverture |
|---|:---:|---|
| **Extraction (APIs)** | 2 | ✅ Météo + Pollution |
| **Chargement (MongoDB)** | 3 | ✅ Upsert, doublons, mise à jour |
| **Transformation (Merge)** | 13 | ✅ Fusion, features, erreurs |
| **Qualité données (Env)** | 22 | ✅ 100% colonnes, nulls, plages, cohérence |
| **Qualité données (Trafic)** | 17 | ✅ 100% colonnes, vitesse, statuts |
| **Preprocessing Kedro (Env)** | 11 | ✅ Features temporelles, flags, déduplication |
| **Preprocessing Kedro (Trafic)** | 13 | ✅ Parsing, normalisation, extraction |
| **ML (XGBoost NO₂)** | 19 | ✅ Entraînement, métriques, prédictions, importance |
| **ML (Comparaison 6 modèles)** | 21 | ✅ Évaluation, cross-validation, JSON, classement |
| **PostgreSQL + MongoDB** | 8 | ✅ Chargement append, extraction, mocks |

</details open>

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
