from app.extract.pollution import fetch_pollution
from app.extract.weather import fetch_weather
from app.load.mongo_loader import save_to_mongo
from app.transform.merge import merge_weather_pollution


class ETLPipeline:
    def __init__(self, weather_extractor, pollution_extractor, merger, loader) -> None:
        self.weather_extractor = weather_extractor
        self.pollution_extractor = pollution_extractor
        self.merger = merger
        self.loader = loader

    def run(self) -> int:
        weather_df = self.weather_extractor()
        pollution_df = self.pollution_extractor()
        merged_df = self.merger(weather_df, pollution_df)
        return self.loader(merged_df)


def run_pipeline() -> int:
    pipeline = ETLPipeline(
        weather_extractor=fetch_weather,
        pollution_extractor=fetch_pollution,
        merger=merge_weather_pollution,
        loader=save_to_mongo,
    )
    return pipeline.run()


if __name__ == "__main__":
    count = run_pipeline()
    print(f"Pipeline completed successfully. Inserted: {count}")
