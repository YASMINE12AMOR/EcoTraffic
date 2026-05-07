import pandas as pd
import mongomock
from unittest.mock import patch

from app.load.mongo_loader import save_to_mongo


@patch("app.load.mongo_loader.MongoClient", new=mongomock.MongoClient)
def test_save_to_mongo(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/")
    monkeypatch.setenv("MONGO_DB", "test_db")
    monkeypatch.setenv("MONGO_COLLECTION", "test_collection")

    df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "temperature_2m": [10, 11],
        "pm10": [20, 22],
    })

    inserted_count = save_to_mongo(df)

    assert inserted_count == 2