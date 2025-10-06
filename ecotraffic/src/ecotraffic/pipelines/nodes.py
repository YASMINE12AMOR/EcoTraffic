import requests
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import os

def fetch_weather(latitude, longitude):
    url_meteo = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}&hourly=temperature_2m,precipitation,wind_speed_10m"
    )
    response = requests.get(url_meteo)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def fetch_pollution(latitude, longitude):
    url_pollution = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={latitude}&longitude={longitude}&hourly=pm10,pm2_5,carbon_monoxide,"
        f"nitrogen_dioxide,ozone,sulphur_dioxide"
    )
    response = requests.get(url_pollution)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["hourly"])
    df.rename(columns={"time": "datetime"}, inplace=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def merge_and_save(weather_data, pollution_data):
    df = pd.merge(weather_data, pollution_data, on="datetime", how="inner")
    
    # MongoDB connection
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    client = MongoClient(mongo_uri)
    db = client["ecotraffic"]
    collection = db["weather_pollution"]
    
    collection.delete_many({})  # facultatif : vider avant d'insérer
    collection.insert_many(df.to_dict("records"))
    print(f"💾 {len(df)} enregistrements insérés dans MongoDB")
