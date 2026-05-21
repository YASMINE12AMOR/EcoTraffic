import pandas as pd
import pytest
from app.transform.merge import merge_weather_pollution


def _make_weather(datetimes=None):
    if datetimes is None:
        datetimes = ["2026-01-01 00:00", "2026-01-01 01:00"]
    return pd.DataFrame({
        "datetime": pd.to_datetime(datetimes),
        "temperature_2m": [10.0] * len(datetimes),
        "precipitation": [0.0] * len(datetimes),
        "wind_speed_10m": [4.0] * len(datetimes),
    })


def _make_pollution(datetimes=None):
    if datetimes is None:
        datetimes = ["2026-01-01 00:00", "2026-01-01 01:00"]
    return pd.DataFrame({
        "datetime": pd.to_datetime(datetimes),
        "pm10": [20.0] * len(datetimes),
        "pm2_5": [10.0] * len(datetimes),
        "nitrogen_dioxide": [7.0] * len(datetimes),
    })


# --- cas nominal ---

def test_merge_returns_non_empty():
    result = merge_weather_pollution(_make_weather(), _make_pollution())
    assert not result.empty


def test_merge_adds_temporal_features():
    result = merge_weather_pollution(_make_weather(), _make_pollution())
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "is_weekend" in result.columns


def test_merge_adds_wind_kmh():
    weather = _make_weather()
    weather["wind_speed_10m"] = [5.0, 10.0]
    result = merge_weather_pollution(weather, _make_pollution())
    assert "wind_kmh" in result.columns
    assert round(result["wind_kmh"].iloc[0], 2) == 18.0


def test_merge_adds_high_pm25_flag():
    pollution = _make_pollution()
    pollution["pm2_5"] = [30.0, 10.0]
    result = merge_weather_pollution(_make_weather(), pollution)
    assert result["high_pm2_5"].iloc[0] == True
    assert result["high_pm2_5"].iloc[1] == False


def test_merge_adds_high_pm10_flag():
    pollution = _make_pollution()
    pollution["pm10"] = [60.0, 20.0]
    result = merge_weather_pollution(_make_weather(), pollution)
    assert result["high_pm10"].iloc[0] == True
    assert result["high_pm10"].iloc[1] == False


def test_merge_is_weekend_flag():
    # 2026-01-03 = samedi, 2026-01-01 = jeudi
    weather = _make_weather(["2026-01-01 00:00", "2026-01-03 00:00"])
    pollution = _make_pollution(["2026-01-01 00:00", "2026-01-03 00:00"])
    result = merge_weather_pollution(weather, pollution)
    assert result.loc[result["datetime"].dt.date == pd.Timestamp("2026-01-01").date(), "is_weekend"].values[0] == False
    assert result.loc[result["datetime"].dt.date == pd.Timestamp("2026-01-03").date(), "is_weekend"].values[0] == True


def test_merge_inner_join_drops_non_matching():
    weather = _make_weather(["2026-01-01 00:00", "2026-01-01 01:00"])
    pollution = _make_pollution(["2026-01-01 00:00"])  # seul 00:00 correspond
    result = merge_weather_pollution(weather, pollution)
    assert len(result) == 1


def test_merge_result_sorted_by_datetime():
    weather = _make_weather(["2026-01-01 02:00", "2026-01-01 00:00"])
    pollution = _make_pollution(["2026-01-01 02:00", "2026-01-01 00:00"])
    result = merge_weather_pollution(weather, pollution)
    assert list(result["datetime"]) == sorted(result["datetime"].tolist())


# --- cas d'erreur ---

def test_merge_raises_if_weather_empty():
    with pytest.raises(ValueError, match="weather_df is empty"):
        merge_weather_pollution(pd.DataFrame(), _make_pollution())


def test_merge_raises_if_pollution_empty():
    with pytest.raises(ValueError, match="pollution_df is empty"):
        merge_weather_pollution(_make_weather(), pd.DataFrame())


def test_merge_raises_if_weather_missing_datetime():
    weather_no_dt = pd.DataFrame({"temperature_2m": [10.0]})
    with pytest.raises(ValueError, match="datetime"):
        merge_weather_pollution(weather_no_dt, _make_pollution())


def test_merge_raises_if_pollution_missing_datetime():
    pollution_no_dt = pd.DataFrame({"pm10": [20.0]})
    with pytest.raises(ValueError, match="datetime"):
        merge_weather_pollution(_make_weather(), pollution_no_dt)