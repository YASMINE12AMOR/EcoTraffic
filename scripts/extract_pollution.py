import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def fetch_pollution():
    url = os.getenv("AIR_QUALITY_API_URL")
    latitude = os.getenv("LATITUDE")
    longitude = os.getenv("LONGITUDE")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": (
            "pm10,pm2_5,carbon_monoxide,"
            "nitrogen_dioxide,ozone,sulphur_dioxide"
        )
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    df = pd.DataFrame(response.json()["hourly"])
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])

    return df
