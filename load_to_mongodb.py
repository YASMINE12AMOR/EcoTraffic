import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

# On réutilise ta classe existante
from extract_api import TrafficAPIClient


class MongoTrafficRepository:
    """
    Gère la connexion MongoDB + insertion des documents trafic.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        load_dotenv()

        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI")
        if not self.mongo_uri:
            raise RuntimeError("MONGODB_URI is not set in .env")

        self.db_name = db_name or os.getenv("MONGO_DB", "traffic_db")
        self.collection_name = collection_name or os.getenv("MONGO_COLLECTION", "rennes_traffic_raw")

        self.client = MongoClient(self.mongo_uri)
        self.collection = self.client[self.db_name][self.collection_name]

        # Évite les doublons (datetime + route)
        self.collection.create_index(
            [("datetime", ASCENDING), ("route", ASCENDING)],
            unique=True
        )

    def insert_dataframe(self, df: pd.DataFrame) -> int:
        """
        Insère un DataFrame dans Mongo. Retourne le nombre de documents insérés.
        """
        docs = df.to_dict(orient="records")
        if not docs:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        for d in docs:
            d["ingested_at"] = now

        try:
            result = self.collection.insert_many(docs, ordered=False)
            return len(result.inserted_ids)
        except Exception as e:
            # En cas de duplicates (index unique) insert_many peut lever une BulkWriteError
            print(f"⚠️ Some documents may already exist: {e}")
            return 0


class TrafficIngestionJob:
    """
    Orchestrateur : extrait depuis l'API et charge dans MongoDB.
    """

    def __init__(
        self,
        api_client: TrafficAPIClient,
        repository: MongoTrafficRepository
    ):
        self.api_client = api_client
        self.repository = repository

    def run(self, rows: int = 100) -> int:
        df = self.api_client.fetch_traffic(rows=rows)
        return self.repository.insert_dataframe(df)


if __name__ == "__main__":
    api_client = TrafficAPIClient()
    repo = MongoTrafficRepository()
    job = TrafficIngestionJob(api_client, repo)

    inserted = job.run(rows=100)
    print(f"✅ Inserted {inserted} documents into {repo.db_name}.{repo.collection_name}")
