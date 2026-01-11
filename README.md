# EcoTraffic 🚦

> Real-time traffic data ingestion and analysis for Rennes Métropole

## 📊 Description

EcoTraffic is a data ingestion script that retrieves and structures real-time road traffic data from the Rennes Métropole Open Data API. The output is a structured Pandas DataFrame where each row represents a road segment observation at a specific point in time.

## 📋 DataFrame Structure

The script outputs a Pandas DataFrame with the following columns:

### 🕒 Date/Heure
Exact date and time of the traffic measurement.

### 🛣️ Route
Name of the road segment (e.g., "Route départementale 34").

### 🚗 Vitesse Moyenne (km/h)
Average speed of vehicles on the segment.
- **Primary indicator** of traffic fluidity

### ⚡ Vitesse Max (km/h)
Maximum observed speed on the segment.

### ⏱️ Temps de Trajet (s)
Estimated travel time to cross the segment (in seconds).

### 🎯 Fiabilité (%)
Reliability level of the travel time estimation.
- A value of `0` indicates low or unavailable reliability

### 🚦 Statut Trafic
Overall traffic status:
- `freeFlow`: Fluid traffic
- `unknown`: Undetermined status

### 🗺️ Hiérarchie


## Kedro : Définition
Kedro est un framework open-source développé par QuantumBlack (McKinsey) pour créer des pipelines de data science reproductibles, maintenables et modulaires.

## 🎯 Avantages principaux :

Structure projet standardisée (comme Django pour le web)
Pipelines modulaires : découper votre code en étapes réutilisables
Gestion des données : catalogues pour sources de données
Versioning : suivi des données et des modèles
Reproductibilité : mêmes résultats à chaque exécution
Visualisation : graphe de dépendances entre les étapes

## 🚀 Getting Started


## 📦 Requirements



## 📄 License

This project uses Open Data from Rennes Métropole.

## 🤝 Contributing



Made with ❤️ for sustainable urban mobility