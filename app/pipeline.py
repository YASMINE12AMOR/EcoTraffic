from app.extract.weather import fetch_weather
from app.extract.pollution import fetch_pollution
from app.transform.merge import merge_weather_pollution
from app.load.mongo_loader import save_to_mongo


def run_pipeline() -> int:
    weather_df = fetch_weather()
    pollution_df = fetch_pollution()
    merged_df = merge_weather_pollution(weather_df, pollution_df)
    inserted_count = save_to_mongo(merged_df)
    return inserted_count


if __name__ == "__main__":
    count = run_pipeline()
    print(f"Pipeline completed successfully. Inserted: {count}")