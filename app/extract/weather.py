import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class ApiExtractor:
    def __init__(self, url: str, latitude: str, longitude: str) -> None:
        self.url = url
        self.latitude = latitude
        self.longitude = longitude

    def _validate_config(self, url_env_name: str) -> None:
        if not self.url:
            raise ValueError(f"{url_env_name} is missing in .env")
        if not self.latitude or not self.longitude:
            raise ValueError("LATITUDE or LONGITUDE is missing in .env")

    def fetch_with_retry(
        self,
        params: dict,
        retries: int = 5,
        initial_delay: int = 5,
        timeout: int = 30,
    ) -> dict:
        headers = {"User-Agent": "EcoTraffic/1.0"}
        delay = initial_delay
        last_error = None

        for attempt in range(1, retries + 1):
            try:
                response = requests.get(
                    self.url,
                    params=params,
                    timeout=timeout,
                    headers=headers,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as error:
                last_error = error
                print(f"Tentative {attempt}/{retries} echouee: {error}")
                if attempt < retries:
                    time.sleep(delay)
                    delay *= 2

        raise RuntimeError(f"API unavailable after {retries} attempts") from last_error


class WeatherExtractor(ApiExtractor):
    def fetch(self) -> pd.DataFrame:
        self._validate_config("WEATHER_API_URL")

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "past_days": 30,
        }

        data = self.fetch_with_retry(params)

        if "hourly" not in data:
            raise ValueError("Invalid weather API response: 'hourly' field not found")

        df = pd.DataFrame(data["hourly"])
        if "time" not in df.columns:
            raise ValueError("Invalid weather API response: 'time' column not found")

        df = df.rename(columns={"time": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df.dropna(subset=["datetime"])


def fetch_weather() -> pd.DataFrame:
    extractor = WeatherExtractor(
        url=os.getenv("WEATHER_API_URL"),
        latitude=os.getenv("LATITUDE"),
        longitude=os.getenv("LONGITUDE"),
    )
    return extractor.fetch()
