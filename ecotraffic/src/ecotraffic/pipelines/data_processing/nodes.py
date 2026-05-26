import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient


DATE_COL = "Date/Heure"
ROUTE_COL = "Route"
AVG_SPEED_COL = "Vitesse Moyenne (km/h)"
MAX_SPEED_COL = "Vitesse Max (km/h)"
TRAVEL_TIME_COL = "Temps de Trajet (s)"
RELIABILITY_COL = "Fiabilit\u00e9 (%)"
STATUS_COL = "Statut Trafic"
HIERARCHY_COL = "Hi\u00e9rarchie"

MOJIBAKE_RELIABILITY_COL = "Fiabilit\u00c3\u00a9 (%)"
MOJIBAKE_HIERARCHY_COL = "Hi\u00c3\u00a9rarchie"
OUTPUT_COLUMNS = [
    DATE_COL,
    ROUTE_COL,
    AVG_SPEED_COL,
    MAX_SPEED_COL,
    TRAVEL_TIME_COL,
    RELIABILITY_COL,
    STATUS_COL,
    HIERARCHY_COL,
    "_id",
    "Latitude",
    "Longitude",
]


def _first_present(fields: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in fields and fields[key] is not None:
            return fields[key]
    return None


def _serialise_mongo_document(document: Dict[str, Any]) -> Dict[str, Any]:
    serialised = dict(document)
    if serialised.get("_id") is not None:
        serialised["_id"] = str(serialised["_id"])
    return serialised

def fetch_traffic_data_from_mongodb(
    connection_string: str,
    database_name: str,
    collection_name: str,
    query: Dict[str, Any] = None,
    limit: int = None
) -> List[Dict[str, Any]]:
    """
    Récupère les données de trafic depuis MongoDB Atlas.
    
    Args:
        connection_string: URI de connexion MongoDB Atlas
        database_name: Nom de la base de données
        collection_name: Nom de la collection
        query: Filtre de requête MongoDB (optionnel)
        limit: Nombre maximum de documents à récupérer (optionnel)
    
    Returns:
        Liste de documents MongoDB
    """
    load_dotenv()

    connection_string = os.getenv("MONGODB_URI", connection_string)
    database_name = os.getenv("MONGO_DB", database_name)
    collection_name = os.getenv("MONGO_COLLECTION", collection_name)

    client = MongoClient(connection_string)
    db = client[database_name]
    collection = db[collection_name]
    
    cursor = collection.find(query or {})
    if limit:
        cursor = cursor.limit(limit)
    
    records = [_serialise_mongo_document(document) for document in cursor]
    client.close()

    if not records:
        raise ValueError(
            "No traffic records were found in MongoDB. "
            "Run `load_to_mongodb.py` first or verify "
            "`MONGODB_URI`, `MONGO_DB`, and `MONGO_COLLECTION`."
        )

    return records

def parse_traffic_records(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []

    for doc in raw_data:
        # 1) si "fields" existe, on l'utilise (structure API type records/fields)
        if isinstance(doc.get("fields"), dict) and doc["fields"]:
            fields = doc["fields"]
        else:
            # 2) sinon, les champs sont directement dans le document
            fields = doc

        # Supporte plusieurs noms possibles selon ton ingestion.
        dt = _first_present(fields, "datetime", DATE_COL)
        route = _first_present(fields, "denomination", "route", ROUTE_COL)
        avg_speed = _first_present(fields, "averagevehiclespeed", "avg_vehicle_speed_kmh", AVG_SPEED_COL)
        max_speed = _first_present(fields, "vitesse_maxi", "max_speed_kmh", MAX_SPEED_COL)
        travel_time = _first_present(fields, "traveltime", "travel_time_s", TRAVEL_TIME_COL)
        reliability = _first_present(
            fields,
            "traveltimereliability",
            "reliability_pct",
            RELIABILITY_COL,
            MOJIBAKE_RELIABILITY_COL,
        )
        status = _first_present(fields, "trafficstatus", "traffic_status", STATUS_COL)
        hierarchie = _first_present(fields, "hierarchie", HIERARCHY_COL, MOJIBAKE_HIERARCHY_COL)

        geo = _first_present(fields, "geo_point_2d") or {}
        lat = geo.get("lat") if isinstance(geo, dict) else _first_present(fields, "latitude", "Latitude")
        lon = geo.get("lon") if isinstance(geo, dict) else _first_present(fields, "longitude", "Longitude")

        records.append({
            DATE_COL: dt,
            ROUTE_COL: route,
            AVG_SPEED_COL: avg_speed,
            MAX_SPEED_COL: max_speed,
            TRAVEL_TIME_COL: travel_time,
            RELIABILITY_COL: reliability,
            STATUS_COL: status,
            HIERARCHY_COL: hierarchie,
            "_id": str(doc.get("_id")) if doc.get("_id") is not None else None,
            "Latitude": lat,
            "Longitude": lon,
        })

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            MOJIBAKE_RELIABILITY_COL: RELIABILITY_COL,
            MOJIBAKE_HIERARCHY_COL: HIERARCHY_COL,
        }
    )


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
    df = _normalise_columns(df.copy())

    # Types
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    numeric_cols = [
        AVG_SPEED_COL,
        MAX_SPEED_COL,
        TRAVEL_TIME_COL,
        RELIABILITY_COL,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Valeurs manquantes (fiabilité)
    df[RELIABILITY_COL] = df[RELIABILITY_COL].fillna(0)

    # Suppression lignes inutilisables
    df = df.dropna(subset=[DATE_COL, AVG_SPEED_COL, TRAVEL_TIME_COL])

    # Doublons stricts
    df = df.drop_duplicates()

    # Normalisation (statut trafic)
    df[STATUS_COL] = df[STATUS_COL].astype(str).str.lower().str.strip()

    # Filtrage valeurs aberrantes (soft)
    df = df[df[AVG_SPEED_COL] <= 130]

    # Trier par date décroissante (comme ton ancienne version)
    df = df.sort_values(DATE_COL, ascending=False)

    # Clé de jointure avec les données climat (tronquée à l'heure)
    df["datetime_hour"] = df[DATE_COL].dt.floor("h")

    return df
