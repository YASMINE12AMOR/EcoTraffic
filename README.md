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
| ML | **XGBoost** + scikit-learn + matplotlib | Modèle NO2 entraîné et visualisé |

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
├── ml/                               # Machine Learning
│   ├── env_model.py                  # Modèle XGBoost — prédiction NO2
│   ├── env_model_viz.py              # Visualisation des résultats
│   └── models/
│       ├── env_no2_model.pkl         # Modèle entraîné (sauvegardé)
│       └── env_model_results.png     # Graphiques des résultats
│
├── tests/                            # Tests automatiques
│   ├── test_data_traffic.py          # Qualité données trafic réelles
│   ├── test_data_env.py              # Qualité données météo+pollution réelles
│   ├── test_env_model.py             # Tests du modèle ML (19 tests)
│   ├── test_merge.py                 # Tests fusion météo+pollution
│   ├── test_traffic_nodes.py         # Tests pipeline trafic Kedro
│   ├── test_env_nodes.py             # Tests pipeline environnement Kedro
│   ├── test_extract.py               # Tests extraction APIs
│   └── test_load.py                  # Tests chargement MongoDB
│
├── .env                              # Variables d'environnement
├── pytest.ini                        # Configuration pytest
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

**XGBoost a été choisi pour 5 raisons :**

| Raison | Explication |
|---|---|
| **Petit dataset** | 120 lignes seulement. XGBoost intègre une régularisation L1/L2 qui évite l'overfitting. Les réseaux de neurones (LSTM) nécessitent des milliers d'exemples. |
| **Features hétérogènes** | Le dataset mélange continu (température, vent), entiers (heure, jour) et binaires (rain_flag). XGBoost ne nécessite aucune normalisation, contrairement à la régression linéaire ou SVR. |
| **Relations non-linéaires** | Le NO2 ne varie pas linéairement avec l'heure (pics aux heures de pointe) ni avec la pluie. XGBoost capture ces interactions automatiquement via ses arbres de décision. |
| **Interprétabilité** | Fournit nativement l'importance des features (CO = 55%, ozone = 34%), essentiel pour justifier les prédictions dans un projet environnemental. |
| **Rapidité** | Entraînement en < 1 seconde, ce qui permet de ré-entraîner à chaque nouvelle collecte sans coût computationnel. |

**Alternatives écartées :**

| Modèle | Raison du rejet |
|---|---|
| Régression linéaire | Relations non-linéaires dans les données → sous-ajustement |
| Random Forest | Moins efficace que XGBoost sur petits datasets (variance élevée) |
| LSTM | Nécessite beaucoup plus de données temporelles séquentielles |
| SVR | Sensible à la normalisation et plus long à calibrer |

### Features utilisées

| Feature | Rôle |
|---|---|
| `carbon_monoxide` | Même source que NO2 (combustion) — 55.5% d'importance |
| `ozone` | Réaction chimique directe avec NO2 — 34.4% |
| `hour` | Heures de pointe vs nuit |
| `day_of_week` | Lundi (rush) vs dimanche |
| `temperature_2m` | Dispersion verticale des polluants |
| `wind_speed_10m` | Dispersion horizontale |
| `precipitation` / `rain_flag` | Nettoyage de l'air par la pluie |

### Résultats

| Métrique | Valeur | Interprétation |
|---|---|---|
| **R²** | **0.758** | Le modèle explique 75.8% de la variance du NO2 |
| **MAE** | 0.587 µg/m³ | Erreur moyenne de 0.6 µg/m³ |
| **RMSE** | 0.886 µg/m³ | Erreur quadratique |

### Fichiers

| Fichier | Rôle |
|---|---|
| `ml/env_model.py` | Entraînement, évaluation, prédiction |
| `ml/env_model_viz.py` | Visualisation des résultats (4 graphiques) |
| `ml/models/env_no2_model.pkl` | Modèle sérialisé |
| `ml/models/env_model_results.png` | Graphiques sauvegardés |

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
| Stockage MongoDB | Fait |
| Stockage PostgreSQL | Fait |
| DAG Airflow trafic (@hourly) | Fait |
| DAG Airflow environnement (@daily) | Fait |
| Tests automatiques (99 tests) | Fait |
| Conteneurisation Docker | Fait |
| **Modèle ML environnement (NO2)** | **Fait** |
| Fusion trafic + environnement | En cours |
| Modèle ML trafic | A faire |
| EcoTraffic Score combiné | A faire |
| Dashboard de visualisation | A faire |

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

### Modèle ML — Prédiction NO2

```bash
# Entraîner le modèle et afficher les métriques
python ml/env_model.py

# Visualiser les résultats (graphiques)
python ml/env_model_viz.py
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

Résultat attendu : **99 passed**

---

### Couverture des tests

| Fichier | Tests | Ce qui est testé |
|---|---|---|
| `tests/test_data_traffic.py` | 17 | Qualité des données réelles trafic (traffic_cleaned.csv) |
| `tests/test_data_env.py` | 22 | Qualité des données réelles météo + pollution (preprocessed_env_df.csv) |
| `tests/test_env_model.py` | 19 | Modèle ML : chargement, entraînement, métriques, prédiction, sauvegarde |
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
