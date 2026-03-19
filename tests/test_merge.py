import pandas as pd
from app.transform.merge import merge_weather_pollution


def test_merge_weather_pollution():
    weather_df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "temperature_2m": [10, 11],
        "precipitation": [0.0, 0.2],
        "wind_speed_10m": [3.0, 4.0],
    })

    pollution_df = pd.DataFrame({
        "datetime": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "pm10": [18, 20],
        "pm2_5": [9, 11],
        "carbon_monoxide": [100, 110],
        "nitrogen_dioxide": [7, 8],
        "ozone": [25, 27],
        "sulphur_dioxide": [1, 2],
    })

    result = merge_weather_pollution(weather_df, pollution_df)

    assert not result.empty
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert "wind_kmh" in result.columns
    assert "high_pm2_5" in result.columns