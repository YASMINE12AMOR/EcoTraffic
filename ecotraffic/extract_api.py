import os
import requests
import pandas as pd
from dotenv import load_dotenv


class TrafficAPIClient:
    """
    Client pour récupérer les données de trafic en temps réel
    depuis l'API OpenData.
    """

    def __init__(self, api_base_url: str | None = None, timeout: int = 30):
        # Charge automatiquement les variables du fichier .env
        load_dotenv()

        self.api_base_url = api_base_url or os.getenv("API_BASE_URL")
        if not self.api_base_url:
            raise RuntimeError("API_BASE_URL is not set in .env")

        self.timeout = timeout

    def fetch_traffic(self, rows: int = 100) -> pd.DataFrame:
        """
        Récupère les données de trafic et retourne un DataFrame pandas.
        """
        params = {
            "dataset": "etat-du-trafic-en-temps-reel",
            "rows": rows,
            "sort": "-datetime",
        }

        response = requests.get(
            self.api_base_url,
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()

        data = response.json()
        return self._parse_records(data)

    @staticmethod
    def _parse_records(data: dict) -> pd.DataFrame:
        """
        Transforme la réponse JSON de l'API en DataFrame.
        """
        records = []

        for rec in data.get("records", []):
            fields = rec.get("fields", {})
            geo = fields.get("geo_point_2d") or {}
            records.append({
                "datetime": fields.get("datetime"),
                "route": fields.get("denomination"),
                "avg_vehicle_speed_kmh": fields.get("averagevehiclespeed"),
                "max_speed_kmh": fields.get("vitesse_maxi"),
                "travel_time_s": fields.get("traveltime"),
                "reliability_pct": fields.get("traveltimereliability"),
                "traffic_status": fields.get("trafficstatus"),
                "hierarchie": fields.get("hierarchie"),
                "latitude": geo.get("lat") if isinstance(geo, dict) else None,
                "longitude": geo.get("lon") if isinstance(geo, dict) else None,
            })

        return pd.DataFrame(records)


if __name__ == "__main__":
    client = TrafficAPIClient()
    df = client.fetch_traffic(rows=100)
    print(df.head(20))
