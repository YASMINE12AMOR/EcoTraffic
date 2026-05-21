import pandas as pd
import pytest
from ecotraffic.pipelines.data_processing.nodes import (
    parse_traffic_records,
    clean_traffic_data,
    _first_present,
)


# --- _first_present ---

def test_first_present_returns_first_match():
    fields = {"a": 1, "b": 2}
    assert _first_present(fields, "a", "b") == 1


def test_first_present_skips_none():
    fields = {"a": None, "b": 5}
    assert _first_present(fields, "a", "b") == 5


def test_first_present_returns_none_if_no_match():
    fields = {"x": 1}
    assert _first_present(fields, "a", "b") is None


# --- parse_traffic_records ---

def test_parse_flat_document():
    raw = [{
        "datetime": "2026-05-18 23:55:00",
        "denomination": "Boulevard Volney",
        "averagevehiclespeed": 50,
        "vitesse_maxi": 50,
        "traveltime": 12,
        "traveltimereliability": 0,
        "trafficstatus": "unknown",
        "hierarchie": "Réseau d'appui",
        "_id": "abc123",
    }]
    df = parse_traffic_records(raw)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["Route"].iloc[0] == "Boulevard Volney"
    assert df["Vitesse Moyenne (km/h)"].iloc[0] == 50


def test_parse_fields_subdocument():
    raw = [{
        "_id": "abc123",
        "fields": {
            "datetime": "2026-05-18 23:55:00",
            "denomination": "Avenue des Préales",
            "averagevehiclespeed": 34,
            "vitesse_maxi": 50,
            "traveltime": 23,
            "traveltimereliability": 60,
            "trafficstatus": "heavy",
            "hierarchie": "Réseau de distribution principale",
        }
    }]
    df = parse_traffic_records(raw)
    assert df["Route"].iloc[0] == "Avenue des Préales"
    assert df["Statut Trafic"].iloc[0] == "heavy"


def test_parse_returns_all_output_columns():
    raw = [{"datetime": "2026-05-18 23:55:00", "_id": "x"}]
    df = parse_traffic_records(raw)
    expected = ["Date/Heure", "Route", "Vitesse Moyenne (km/h)", "Vitesse Max (km/h)",
                "Temps de Trajet (s)", "Fiabilité (%)", "Statut Trafic", "Hiérarchie", "_id"]
    for col in expected:
        assert col in df.columns


def test_parse_missing_fields_returns_none():
    raw = [{"_id": "x"}]
    df = parse_traffic_records(raw)
    assert df["Route"].iloc[0] is None


# --- clean_traffic_data ---

def _sample_df():
    return pd.DataFrame({
        "Date/Heure": ["2026-05-18 23:55:00+02:00", "2026-05-18 23:30:00+02:00", "2026-05-18 22:00:00+02:00"],
        "Route": ["Boulevard Volney", "Avenue des Préales", "Avenue des Buttes"],
        "Vitesse Moyenne (km/h)": [50, 34, 45],
        "Vitesse Max (km/h)": [50, 50, 50],
        "Temps de Trajet (s)": [12, 23, 33],
        "Fiabilité (%)": [0.0, 60.0, 60.0],
        "Statut Trafic": ["Unknown", "HEAVY", "freeflow"],
        "Hiérarchie": ["Réseau d'appui", "Réseau de distribution", "Réseau d'appui"],
        "_id": ["a1", "a2", "a3"],
    })


def test_clean_adds_datetime_hour():
    df = clean_traffic_data(_sample_df())
    assert "datetime_hour" in df.columns


def test_clean_normalizes_status_to_lowercase():
    df = clean_traffic_data(_sample_df())
    assert all(df["Statut Trafic"].str.islower())


def test_clean_removes_speed_above_130():
    raw = _sample_df()
    raw.loc[0, "Vitesse Moyenne (km/h)"] = 150
    df = clean_traffic_data(raw)
    assert (df["Vitesse Moyenne (km/h)"] <= 130).all()


def test_clean_fills_reliability_na():
    raw = _sample_df()
    raw.loc[0, "Fiabilité (%)"] = None
    df = clean_traffic_data(raw)
    assert df["Fiabilité (%)"].isna().sum() == 0


def test_clean_removes_duplicates():
    raw = _sample_df()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    df = clean_traffic_data(raw)
    assert len(df) == len(_sample_df())


def test_clean_drops_rows_with_null_speed():
    raw = _sample_df()
    raw.loc[0, "Vitesse Moyenne (km/h)"] = None
    df = clean_traffic_data(raw)
    assert len(df) == 2


def test_clean_drops_rows_with_null_date():
    raw = _sample_df()
    raw.loc[0, "Date/Heure"] = None
    df = clean_traffic_data(raw)
    assert len(df) == 2
