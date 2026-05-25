import os
import sys
import importlib.util
import json
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import SSLError
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAFFIC_NODES_PATH = (
    ROOT_DIR
    / "ecotraffic"
    / "src"
    / "ecotraffic"
    / "pipelines"
    / "data_processing"
    / "nodes.py"
)
OUTPUT_PATH = ROOT_DIR / "ecotraffic" / "data" / "03_primary" / "traffic_cleaned.csv"


def load_clean_traffic_data():
    spec = importlib.util.spec_from_file_location("traffic_nodes", TRAFFIC_NODES_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load traffic nodes from {TRAFFIC_NODES_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["traffic_nodes"] = module
    spec.loader.exec_module(module)
    return module.clean_traffic_data


clean_traffic_data = load_clean_traffic_data()


def fetch_traffic(rows: int = 1000) -> pd.DataFrame:
    load_dotenv(ROOT_DIR / ".env")

    api_base_url = os.getenv(
        "API_BASE_URL",
        "https://data.rennesmetropole.fr/api/records/1.0/search/",
    )
    params = {
        "dataset": "etat-du-trafic-en-temps-reel",
        "rows": rows,
        "sort": "-datetime",
    }

    try:
        response = requests.get(api_base_url, params=params, timeout=30)
    except SSLError:
        print("SSL certificate verification failed. Retrying with verification disabled for this public API.")
        response = requests.get(api_base_url, params=params, timeout=30, verify=False)
    response.raise_for_status()
    payload = response.json()

    records = []
    for rec in payload.get("records", []):
        fields = rec.get("fields", {})
        geo_point = fields.get("geo_point_2d") or [None, None]
        records.append(
            {
                "Date/Heure": fields.get("datetime"),
                "Route": fields.get("denomination"),
                "Vitesse Moyenne (km/h)": fields.get("averagevehiclespeed"),
                "Vitesse Max (km/h)": fields.get("vitesse_maxi"),
                "Temps de Trajet (s)": fields.get("traveltime"),
                "Fiabilite (%)": fields.get("traveltimereliability"),
                "Statut Trafic": fields.get("trafficstatus"),
                "Hierarchie": fields.get("hierarchie"),
                "Latitude": geo_point[0],
                "Longitude": geo_point[1],
                "Geo Shape": json.dumps(fields.get("geo_shape"), ensure_ascii=False),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    raw_df = fetch_traffic()
    if raw_df.empty:
        raise RuntimeError("No traffic data returned by Rennes Metropole API.")

    raw_df = raw_df.rename(
        columns={
            "Fiabilite (%)": "Fiabilité (%)",
            "Hierarchie": "Hiérarchie",
        }
    )
    cleaned_df = clean_traffic_data(raw_df)
    for optional_column in ["Latitude", "Longitude", "Geo Shape"]:
        if optional_column in raw_df.columns and optional_column not in cleaned_df.columns:
            cleaned_df = cleaned_df.merge(
                raw_df[["Date/Heure", "Route", optional_column]],
                on=["Date/Heure", "Route"],
                how="left",
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Traffic raw rows: {len(raw_df)}")
    print(f"Traffic cleaned rows: {len(cleaned_df)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
