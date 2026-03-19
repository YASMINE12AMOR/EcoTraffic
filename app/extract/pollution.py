import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def fetch_pollution() -> pd.DataFrame:
    url = os.getenv("AIR_QUALITY_API_URL")
    latitude = os.getenv("LATITUDE")
    longitude = os.getenv("LONGITUDE")

    if not url:
        raise ValueError("AIR_QUALITY_API_URL is missing in .env")
    if not latitude or not longitude:
        raise ValueError("LATITUDE or LONGITUDE is missing in .env")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    if "hourly" not in data:
        raise ValueError("Invalid pollution API response: 'hourly' field not found")

    df = pd.DataFrame(data["hourly"])
    if "time" not in df.columns:
        raise ValueError("Invalid pollution API response: 'time' column not found")

    df = df.rename(columns={"time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    return df