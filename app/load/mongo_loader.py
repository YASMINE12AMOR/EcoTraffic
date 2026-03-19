import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def save_to_mongo(df: pd.DataFrame) -> int:
    if df.empty:
        raise ValueError("The DataFrame is empty, nothing to save")

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db = os.getenv("MONGO_DB", "ecotraffic")
    mongo_collection = os.getenv("MONGO_COLLECTION", "weather_pollution")

    if not mongo_uri:
        raise ValueError("MONGO_URI is missing in .env")

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    collection = db[mongo_collection]

    records = df.copy()

    # MongoDB n'aime pas certains types pandas
    for col in records.columns:
        if str(records[col].dtype).startswith("datetime64"):
            records[col] = records[col].astype(str)

    documents = records.to_dict("records")

    collection.delete_many({})
    result = collection.insert_many(documents)

    inserted_count = len(result.inserted_ids)
    print(f"✅ {inserted_count} documents inserted into MongoDB")

    client.close()
    return inserted_count