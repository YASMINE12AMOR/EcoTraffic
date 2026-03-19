# 🚦 EcoTraffic – Pipeline de données (Trafic, Météo & Qualité de l’air)

## 📌 Présentation du projet

**EcoTraffic** est un projet de data engineering qui met en place un **pipeline ETL automatisé** permettant de collecter, transformer et stocker des données environnementales :

* 🌦️ Données météo
* 🌫️ Qualité de l’air

Le pipeline est orchestré avec **Apache Airflow** et les données sont stockées dans **MongoDB Atlas (base NoSQL cloud)**.

---

## 🎯 Objectifs

* Collecter des données en temps réel via des APIs
* Nettoyer et fusionner des données hétérogènes
* Stocker les données dans une base NoSQL
* Automatiser l’exécution du pipeline
* Garantir la qualité avec des tests

---

## 🏗️ Architecture

```
            ┌───────────────┐
            │ API Météo     │
            └──────┬────────┘
                   │
            ┌──────▼────────┐
            │ API Qualité   │
            │ de l’air      │
            └──────┬────────┘
                   │
               [Extract]
                   │
               [Transform]
                   │
               [Load]
                   │
            ┌──────▼────────┐
            │ MongoDB Atlas │
            └──────┬────────┘
                   │
             Apache Airflow
        (Orchestration & planification)
```

---

## ⚙️ Stack technique

* **Python 3**
* **Pandas** → traitement des données
* **MongoDB Atlas** → base NoSQL cloud
* **Apache Airflow** → orchestration
* **Docker** → conteneurisation
* **Pytest** → tests unitaires

### APIs utilisées :

* Open-Meteo (météo)
* Open-Meteo Air Quality

---

## 📂 Structure du projet

```
EcoTraffic/
│
├── app/
│   ├── pipeline.py        # Pipeline ETL principal
│   ├── extract.py         # Extraction des données
│   ├── transform.py       # Transformation 
│   ├── load.py            # Chargement 
│
├── airflow/
│   └── dags/
│       └── etl_orchestration.py
│
├── tests/
│   └── test_pipeline.py
│
├── .env
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔄 Fonctionnement du pipeline ETL

### 1️⃣ Extract (Extraction)

* Récupération des données météo via API
* Récupération des données de qualité de l’air

---

### 2️⃣ Transform (Transformation)

* Conversion des dates (datetime)
* Nettoyage des données
* Fusion des datasets

---

### 3️⃣ Load (Chargement)

* Insertion dans MongoDB Atlas
* Collection : `weather_pollution`

---

## ▶️ Lancer le pipeline en local

```bash id="run123"
python -m app.pipeline
```

Résultat attendu :

```id="out123"
Inserted XXX documents into MongoDB
```

---

## 🐳 Lancer avec Docker & Airflow

```bash id="docker123"
docker compose up --build
```

Accès Airflow :

```id="airflow123"
http://localhost:8080
```

Identifiants :

* Username : `admin`
* Password : `admin123`

---

## ⏰ Automatisation avec Airflow

Le pipeline est exécuté automatiquement :

```python id="schedule123"
schedule_interval="@daily"
```

Airflow permet :

* la planification
* le monitoring
* la gestion des logs
* la reprise en cas d’erreur

---

## 🧪 Tests

Lancer les tests :

```bash id="test123"
pytest
```

Tests réalisés :

* extraction des données
* transformation
* pipeline complet

---

## 🌍 Variables d’environnement (.env)

```env id="env123"
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/ecotraffic
MONGO_DB=ecotraffic
MONGO_COLLECTION=weather_pollution

WEATHER_API_URL=https://api.open-meteo.com/v1/forecast
AIR_QUALITY_API_URL=https://air-quality-api.open-meteo.com/v1/air-quality

LATITUDE=48.1173
LONGITUDE=-1.6778
```

---

## 📊 Améliorations possibles

* Intégration des données de trafic 🚗
* Création d’un dashboard (Streamlit / Power BI)
* Ajout de modèles prédictifs
* Déploiement cloud (GCP / AWS)

---

## 💡 Compétences développées

* Conception d’un pipeline ETL complet
* Utilisation d’APIs temps réel
* Orchestration avec Airflow
* Gestion de bases NoSQL
* Mise en place de tests

---

## 👩‍💻 Auteur

**Eya Ben Salem**
Master Big Data & IA

---

## ⭐ Conclusion

Ce projet démontre la mise en place d’un **pipeline de données automatisé**, proche des standards professionnels, combinant ingestion, transformation, stockage et orchestration.

---

✨ *Projet réutilisable et extensible pour des cas d’usage data engineering réels.*
