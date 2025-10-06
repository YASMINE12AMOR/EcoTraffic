from kedro.pipeline import Pipeline, node
from .nodes import fetch_weather, fetch_pollution, merge_and_save

def create_pipeline(**kwargs):
    latitude = 48.1173
    longitude = -1.6778
    return Pipeline(
        [
            node(fetch_weather, inputs=dict(latitude=latitude, longitude=longitude),
                 outputs="weather_data", name="fetch_weather_node"),
            node(fetch_pollution, inputs=dict(latitude=latitude, longitude=longitude),
                 outputs="pollution_data", name="fetch_pollution_node"),
            node(merge_and_save, inputs=["weather_data", "pollution_data"],
                 outputs=None, name="merge_and_save_node"),
        ]
    )
