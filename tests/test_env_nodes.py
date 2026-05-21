import pandas as pd
import pytest
from ecotraffic_kedro.pipelines.preprocessing.nodes import preprocess_env


def _sample_env():
    return pd.DataFrame({
        "_id": ["id1", "id2", "id3"],
        "datetime": [
            "2026-05-18 00:00:00+00:00",
            "2026-05-18 01:00:00+00:00",
            "2026-05-19 00:00:00+00:00",
        ],
        "temperature_2m": [10.8, 10.1, 12.0],
        "precipitation": [0.0, 0.5, 0.0],
        "wind_speed_10m": [4.2, 6.0, 3.0],
        "pm10": [10.3, 10.7, 9.5],
        "pm2_5": [6.6, 7.3, 5.0],
        "carbon_monoxide": [156.0, 145.0, 130.0],
        "nitrogen_dioxide": [7.4, 6.3, 4.0],
        "ozone": [54.0, 51.0, 60.0],
        "sulphur_dioxide": [0.3, 0.3, 0.2],
    })


# --- cas nominal ---

def test_preprocess_drops_id_column():
    df = preprocess_env(_sample_env())
    assert "_id" not in df.columns


def test_preprocess_adds_hour():
    df = preprocess_env(_sample_env())
    assert "hour" in df.columns
    assert df["hour"].iloc[0] == 0
    assert df["hour"].iloc[1] == 1


def test_preprocess_adds_day_of_week():
    df = preprocess_env(_sample_env())
    assert "day_of_week" in df.columns


def test_preprocess_adds_month():
    df = preprocess_env(_sample_env())
    assert "month" in df.columns
    assert (df["month"] == 5).all()


def test_preprocess_adds_is_weekend():
    # 2026-05-18 = lundi (0), 2026-05-23 = samedi (5)
    raw = _sample_env()
    raw.loc[2, "datetime"] = "2026-05-23 00:00:00+00:00"
    df = preprocess_env(raw)
    assert "is_weekend" in df.columns
    assert df[df["hour"] == 0]["is_weekend"].iloc[0] == 0
    assert df.iloc[-1]["is_weekend"] == 1


def test_preprocess_rain_flag_set_when_precipitation():
    df = preprocess_env(_sample_env())
    assert "rain_flag" in df.columns
    # precipitation=0.5 à 01:00 → rain_flag=1
    assert df[df["hour"] == 1]["rain_flag"].values[0] == 1
    # precipitation=0.0 à 00:00 → rain_flag=0
    assert df[df["hour"] == 0]["rain_flag"].values[0] == 0


def test_preprocess_sorts_by_datetime_ascending():
    raw = _sample_env()
    raw = raw.iloc[::-1].reset_index(drop=True)  # inverser l'ordre
    df = preprocess_env(raw)
    assert list(df["datetime"]) == sorted(df["datetime"].tolist())


def test_preprocess_deduplicates_datetime():
    raw = _sample_env()
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    df = preprocess_env(raw)
    assert df["datetime"].nunique() == len(df)


def test_preprocess_numeric_columns_are_float():
    df = preprocess_env(_sample_env())
    for col in ["temperature_2m", "pm10", "pm2_5", "nitrogen_dioxide"]:
        assert pd.api.types.is_float_dtype(df[col])


# --- cas limites ---

def test_preprocess_empty_df_returns_empty():
    df = preprocess_env(pd.DataFrame())
    assert df.empty


def test_preprocess_drops_rows_with_null_datetime():
    raw = _sample_env()
    raw.loc[0, "datetime"] = None
    df = preprocess_env(raw)
    assert len(df) == 2
