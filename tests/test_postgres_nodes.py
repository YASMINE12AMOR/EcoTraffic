"""
Tests des noeuds de chargement PostgreSQL et extraction MongoDB.
Utilise des mocks pour ne pas necessiter de vraie base de donnees.
"""
import pandas as pd
import pytest
import mongomock
from unittest.mock import patch, MagicMock

from ecotraffic_kedro.pipelines.preprocessing.nodes import (
    save_env_to_postgres,
    save_traffic_to_postgres,
    extract_env_from_mongo,
)


def _sample_env_df():
    return pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00+00:00", "2026-01-01 01:00+00:00"]),
        "temperature_2m": [10.0, 11.0],
        "precipitation": [0.0, 0.5],
        "wind_speed_10m": [4.0, 5.0],
        "pm10": [20.0, 22.0],
        "pm2_5": [10.0, 12.0],
        "carbon_monoxide": [140.0, 150.0],
        "nitrogen_dioxide": [7.0, 8.0],
        "ozone": [54.0, 50.0],
        "sulphur_dioxide": [0.3, 0.4],
        "hour": [0, 1],
        "day_of_week": [3, 3],
        "month": [1, 1],
        "is_weekend": [0, 0],
        "rain_flag": [0, 1],
    })


def _sample_traffic_df():
    return pd.DataFrame({
        "Date/Heure": ["2026-05-28T08:00:00+00:00", "2026-05-28T09:00:00+00:00"],
        "Route": ["Avenue de la Gare", "Boulevard de la Liberte"],
        "Vitesse Moyenne (km/h)": [45.0, 30.0],
        "Vitesse Max (km/h)": [70.0, 50.0],
        "Temps de Trajet (s)": [120.0, 200.0],
        "Fiabilite (%)": [85.0, 72.0],
        "Statut Trafic": ["heavy", "congested"],
        "Hierarchie": ["Reseau d'armature", "Reseau de distribution principale"],
        "_id": ["id1", "id2"],
        "datetime_hour": ["2026-05-28T08:00:00+00:00", "2026-05-28T09:00:00+00:00"],
    })


# ── save_env_to_postgres ────────────────────────────────────────────────────

@patch("ecotraffic_kedro.pipelines.preprocessing.nodes.create_engine")
def test_save_env_to_postgres_appelle_to_sql(mock_engine, monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake:fake@localhost/fakedb")
    monkeypatch.setenv("POSTGRES_TABLE_ENV", "environment_features")

    mock_engine_instance = MagicMock()
    mock_engine.return_value = mock_engine_instance

    df = _sample_env_df()
    with patch.object(df.__class__, "to_sql") as mock_to_sql:
        save_env_to_postgres(df)
        mock_to_sql.assert_called_once()


@patch("ecotraffic_kedro.pipelines.preprocessing.nodes.create_engine")
def test_save_env_to_postgres_df_vide_ne_plante_pas(mock_engine, monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake:fake@localhost/fakedb")
    save_env_to_postgres(pd.DataFrame())
    mock_engine.assert_not_called()


@patch("ecotraffic_kedro.pipelines.preprocessing.nodes.create_engine")
def test_save_env_to_postgres_utilise_bonne_table(mock_engine, monkeypatch, capsys):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake:fake@localhost/fakedb")
    monkeypatch.setenv("POSTGRES_TABLE_ENV", "ma_table_env")

    mock_engine.return_value = MagicMock()
    df = _sample_env_df()

    with patch.object(pd.DataFrame, "to_sql"):
        save_env_to_postgres(df)

    captured = capsys.readouterr()
    assert "ma_table_env" in captured.out


# ── save_traffic_to_postgres ────────────────────────────────────────────────

@patch("ecotraffic_kedro.pipelines.preprocessing.nodes.create_engine")
def test_save_traffic_to_postgres_appelle_to_sql(mock_engine, monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake:fake@localhost/fakedb")
    monkeypatch.setenv("POSTGRES_TABLE_TRAFFIC", "traffic_features")

    mock_engine.return_value = MagicMock()
    df = _sample_traffic_df()

    with patch.object(pd.DataFrame, "to_sql") as mock_to_sql:
        save_traffic_to_postgres(df)
        mock_to_sql.assert_called_once()


@patch("ecotraffic_kedro.pipelines.preprocessing.nodes.create_engine")
def test_save_traffic_to_postgres_df_vide_ne_plante_pas(mock_engine, monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake:fake@localhost/fakedb")
    save_traffic_to_postgres(pd.DataFrame())
    mock_engine.assert_not_called()


@patch("ecotraffic_kedro.pipelines.preprocessing.nodes.create_engine")
def test_save_traffic_normalise_statut(mock_engine, monkeypatch):
    monkeypatch.setenv("POSTGRES_URI", "postgresql://fake:fake@localhost/fakedb")
    monkeypatch.setenv("POSTGRES_TABLE_TRAFFIC", "traffic_features")

    mock_engine.return_value = MagicMock()
    df = _sample_traffic_df()
    df["Statut Trafic"] = ["HEAVY", "FREEFLOW"]

    captured = []

    def fake_to_sql(self, name, engine, **kwargs):
        captured.append(self.copy())

    with patch.object(pd.DataFrame, "to_sql", fake_to_sql):
        save_traffic_to_postgres(df)

    assert len(captured) == 1
    assert all(s == s.lower() for s in captured[0]["Statut Trafic"])


# ── extract_env_from_mongo ──────────────────────────────────────────────────

def test_extract_env_from_mongo_retourne_dataframe(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/")
    monkeypatch.setenv("MONGO_DB", "test_db")
    monkeypatch.setenv("MONGO_COLLECTION_ENV", "weather_pollution")

    client = mongomock.MongoClient("mongodb://localhost:27017/")
    client["test_db"]["weather_pollution"].insert_many([
        {"datetime": "2026-01-01 00:00", "temperature_2m": 10.0, "nitrogen_dioxide": 7.0},
        {"datetime": "2026-01-01 01:00", "temperature_2m": 11.0, "nitrogen_dioxide": 8.0},
    ])

    with patch("ecotraffic_kedro.pipelines.preprocessing.nodes.MongoClient", return_value=client):
        df = extract_env_from_mongo()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "temperature_2m" in df.columns


def test_extract_env_from_mongo_collection_vide(monkeypatch):
    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/")
    monkeypatch.setenv("MONGO_DB", "test_db_vide")
    monkeypatch.setenv("MONGO_COLLECTION_ENV", "vide")

    client = mongomock.MongoClient("mongodb://localhost:27017/")

    with patch("ecotraffic_kedro.pipelines.preprocessing.nodes.MongoClient", return_value=client):
        df = extract_env_from_mongo()

    assert isinstance(df, pd.DataFrame)
    assert df.empty
