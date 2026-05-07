import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_weather():
    url = os.getenv("WEATHER_API_URL")
    lat = os.getenv("LATITUDE")
    lon = os.getenv("LONGITUDE")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df
