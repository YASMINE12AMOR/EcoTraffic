import os

import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


class MongoLoader:
    def __init__(self, mongo_uri: str, mongo_db: str, mongo_collection: str) -> None:
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self.mongo_collection = mongo_collection

    def save(self, df: pd.DataFrame) -> int:
        if df.empty:
            raise ValueError("The DataFrame is empty, nothing to save")
        if not self.mongo_uri:
            raise ValueError("MONGO_URI is missing in .env")

        client = MongoClient(self.mongo_uri)
        db = client[self.mongo_db]
        collection = db[self.mongo_collection]

        records = df.copy()
        for col in records.columns:
            if str(records[col].dtype).startswith("datetime64"):
                records[col] = records[col].astype(str)

        documents = records.to_dict("records")

        collection.delete_many({})
        result = collection.insert_many(documents)
        inserted_count = len(result.inserted_ids)
        print(f"Inserted {inserted_count} documents into MongoDB")

        client.close()
        return inserted_count


def save_to_mongo(df: pd.DataFrame) -> int:
    loader = MongoLoader(
        mongo_uri=os.getenv("MONGO_URI"),
        mongo_db=os.getenv("MONGO_DB", "ecotraffic"),
        mongo_collection=os.getenv("MONGO_COLLECTION", "weather_pollution"),
    )
    return loader.save(df)
