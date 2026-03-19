import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def merge_and_save(weather_df, pollution_df):
    df = pd.merge(weather_df, pollution_df, on="datetime", how="inner")

    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)

    db = client["ecotraffic"]
    collection = db["weather_pollution"]

    collection.delete_many({})  # optionnel
    collection.insert_many(df.to_dict("records"))

    print(f"✅ {len(df)} documents insérés dans MongoDB")
