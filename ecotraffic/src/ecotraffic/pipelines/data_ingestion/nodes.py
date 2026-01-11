"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""
import requests
import pandas as pd
from typing import Dict, Any


def fetch_traffic_data(api_url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Récupère les données brutes depuis l'API Rennes Métropole.

    Args:
        api_url: URL de l'API
        params: Paramètres de la requête

    Returns:
        Données JSON brutes (dict)
    """
    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def parse_traffic_records(raw_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse les données JSON et extrait les champs pertinents.

    Args:
        raw_data: Données JSON brutes de l'API

    Returns:
        DataFrame avec les données structurées
    """
    records = []
    for rec in raw_data.get("records", []):
        fields = rec.get("fields", {})
        records.append({
            "Date/Heure": fields.get("datetime"),
            "Route": fields.get("denomination"),
            "Vitesse Moyenne (km/h)": fields.get("averagevehiclespeed"),
            "Vitesse Max (km/h)": fields.get("vitesse_maxi"),
            "Temps de Trajet (s)": fields.get("traveltime"),
            "Fiabilité (%)": fields.get("traveltimereliability"),
            "Statut Trafic": fields.get("trafficstatus"),
            "Hiérarchie": fields.get("hierarchie"),
        })

    return pd.DataFrame(records)


def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et prépare les données pour analyse/modélisation.

    - Convertit Date/Heure en datetime
    - Force les colonnes numériques
    - Gère la fiabilité manquante
    - Supprime les lignes inutilisables
    - Supprime les doublons stricts
    - Normalise le statut trafic
    - Filtre les vitesses aberrantes

    Args:
        df: DataFrame brut

    Returns:
        DataFrame nettoyé
    """
    df = df.copy()

    # Types
    df["Date/Heure"] = pd.to_datetime(df["Date/Heure"], errors="coerce")

    numeric_cols = [
        "Vitesse Moyenne (km/h)",
        "Vitesse Max (km/h)",
        "Temps de Trajet (s)",
        "Fiabilité (%)"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Valeurs manquantes (fiabilité)
    df["Fiabilité (%)"] = df["Fiabilité (%)"].fillna(0)

    # Suppression lignes inutilisables
    df = df.dropna(subset=["Date/Heure", "Vitesse Moyenne (km/h)", "Temps de Trajet (s)"])

    # Doublons stricts
    df = df.drop_duplicates()

    # Normalisation (statut trafic)
    df["Statut Trafic"] = df["Statut Trafic"].astype(str).str.lower().str.strip()

    # Filtrage valeurs aberrantes (soft)
    df = df[df["Vitesse Moyenne (km/h)"] <= 130]

    # Trier par date décroissante (comme ton ancienne version)
    df = df.sort_values("Date/Heure", ascending=False)

    return df
