import os
import pandas as pd
from dotenv import load_dotenv

from app.extract.weather import ApiExtractor

load_dotenv()


class PollutionExtractor(ApiExtractor):
    def fetch(self) -> pd.DataFrame:
        self._validate_config("AIR_QUALITY_API_URL")

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide",
        }

        data = self.fetch_with_retry(params)

        if "hourly" not in data:
            raise ValueError("Invalid pollution API response: 'hourly' field not found")

        df = pd.DataFrame(data["hourly"])
        if "time" not in df.columns:
            raise ValueError("Invalid pollution API response: 'time' column not found")

        df = df.rename(columns={"time": "datetime"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df.dropna(subset=["datetime"])


def fetch_pollution() -> pd.DataFrame:
    extractor = PollutionExtractor(
        url=os.getenv("AIR_QUALITY_API_URL"),
        latitude=os.getenv("LATITUDE"),
        longitude=os.getenv("LONGITUDE"),
    )
    return extractor.fetch()
