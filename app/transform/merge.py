import pandas as pd


def merge_weather_pollution(
    weather_df: pd.DataFrame,
    pollution_df: pd.DataFrame
) -> pd.DataFrame:
    if weather_df.empty:
        raise ValueError("weather_df is empty")
    if pollution_df.empty:
        raise ValueError("pollution_df is empty")

    if "datetime" not in weather_df.columns:
        raise ValueError("weather_df must contain a 'datetime' column")
    if "datetime" not in pollution_df.columns:
        raise ValueError("pollution_df must contain a 'datetime' column")

    df = pd.merge(weather_df, pollution_df, on="datetime", how="inner")
    df = df.sort_values("datetime").reset_index(drop=True)

    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    df["is_weekend"] = df["datetime"].dt.weekday >= 5

    if "wind_speed_10m" in df.columns:
        df["wind_kmh"] = df["wind_speed_10m"] * 3.6

    if "pm2_5" in df.columns:
        df["high_pm2_5"] = df["pm2_5"] > 25

    if "pm10" in df.columns:
        df["high_pm10"] = df["pm10"] > 50

    return df
