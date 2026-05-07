import os
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


ENV_NUMERIC_COLS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "ozone",
    "sulphur_dioxide",
]


def extract_env_from_mongo() -> pd.DataFrame:
    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB")
    mongo_collection = os.getenv("MONGO_COLLECTION")

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db[mongo_collection]

    docs = list(collection.find({}))
    df = pd.DataFrame(docs)

    print(f"Documents MongoDB lus : {len(df)}")
    return df


def preprocess_env(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if df.empty:
        print("Aucune donnée à transformer.")
        return df

    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

    for col in ENV_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime"])
    df = df.drop_duplicates(subset=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["rain_flag"] = (df["precipitation"].fillna(0) > 0).astype(int)

    print(f"Données transformées : {len(df)} lignes")
    return df


def save_env_to_postgres(df: pd.DataFrame) -> None:
    if df.empty:
        print("Aucune donnée à écrire dans PostgreSQL.")
        return

    postgres_uri = os.getenv("POSTGRES_URI")
    postgres_table = os.getenv("POSTGRES_TABLE_ENV", "environment_features")

    engine = create_engine(postgres_uri)
    df.to_sql(postgres_table, engine, if_exists="replace", index=False)

    print(f"{len(df)} lignes écrites dans PostgreSQL -> table {postgres_table}")