import pandas as pd
import mongomock
from unittest.mock import patch

from app.load.mongo_loader import MongoLoader, save_to_mongo


@patch("app.load.mongo_loader.MongoClient", new=mongomock.MongoClient)
def test_save_to_mongo(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/")
    monkeypatch.setenv("MONGO_DB", "test_db")
    monkeypatch.setenv("MONGO_COLLECTION_ENV", "test_collection")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "temperature_2m": [10, 11],
        "pm10": [20, 22],
    })

    upserted_count = save_to_mongo(df)
    assert upserted_count == 2


def test_upsert_no_duplicates():
    client = mongomock.MongoClient()
    db = client["test_upsert_db"]
    collection = db["test_upsert_col"]

    loader = MongoLoader.__new__(MongoLoader)
    loader.mongo_uri = "mongodb://localhost:27017/"
    loader.mongo_db = "test_upsert_db"
    loader.mongo_collection = "test_upsert_col"

    df1 = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "temperature_2m": [10, 11],
    })
    df2 = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 01:00", "2026-01-01 02:00"]),
        "temperature_2m": [12, 13],
    })

    with patch("app.load.mongo_loader.MongoClient", return_value=client):
        loader.save(df1)
        loader.save(df2)

    assert collection.count_documents({}) == 3


def test_upsert_updates_existing():
    client = mongomock.MongoClient()
    db = client["test_update_db"]
    collection = db["test_update_col"]

    loader = MongoLoader.__new__(MongoLoader)
    loader.mongo_uri = "mongodb://localhost:27017/"
    loader.mongo_db = "test_update_db"
    loader.mongo_collection = "test_update_col"

    df1 = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00"]),
        "temperature_2m": [10],
    })
    df2 = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00"]),
        "temperature_2m": [99],
    })

    with patch("app.load.mongo_loader.MongoClient", return_value=client):
        loader.save(df1)
        loader.save(df2)

    assert collection.count_documents({}) == 1
    doc = list(collection.find({}))[0]
    assert doc["temperature_2m"] == 99