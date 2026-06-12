import pandas as pd
from unittest.mock import patch, Mock

from app.extract.weather import fetch_weather
from app.extract.pollution import fetch_pollution


@patch("app.extract.weather.requests.get")
def test_fetch_weather(mock_get, monkeypatch):
    monkeypatch.setenv("WEATHER_API_URL", "https://fake-weather-api.com")
    monkeypatch.setenv("LATITUDE", "48.1173")
    monkeypatch.setenv("LONGITUDE", "-1.6778")

    fake_response = Mock()
    fake_response.json.return_value = {
        "hourly": {
            "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
            "temperature_2m": [12.3, 11.8],
            "precipitation": [0.0, 0.1],
            "wind_speed_10m": [5.0, 4.5],
        }
    }
    fake_response.raise_for_status.return_value = None
    mock_get.return_value = fake_response

    df = fetch_weather()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "datetime" in df.columns
    assert "temperature_2m" in df.columns


@patch("app.extract.weather.requests.get")
def test_fetch_pollution(mock_get, monkeypatch):
    monkeypatch.setenv("AIR_QUALITY_API_URL", "https://fake-pollution-api.com")
    monkeypatch.setenv("LATITUDE", "48.1173")
    monkeypatch.setenv("LONGITUDE", "-1.6778")

    fake_response = Mock()
    fake_response.json.return_value = {
        "hourly": {
            "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
            "pm10": [20, 22],
            "pm2_5": [12, 14],
            "carbon_monoxide": [100, 110],
            "nitrogen_dioxide": [8, 9],
            "ozone": [30, 29],
            "sulphur_dioxide": [2, 2],
        }
    }
    fake_response.raise_for_status.return_value = None
    mock_get.return_value = fake_response

    df = fetch_pollution()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "datetime" in df.columns
    assert "pm10" in df.columns