"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""
from kedro.pipeline import Pipeline, node
from .nodes import fetch_traffic_data, parse_traffic_records, clean_traffic_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=fetch_traffic_data,
                inputs=["params:api_url", "params:api_params"],
                outputs="raw_traffic_data",
                name="fetch_traffic_data_node",
            ),
            node(
                func=parse_traffic_records,
                inputs="raw_traffic_data",
                outputs="parsed_traffic_data",
                name="parse_traffic_records_node",
            ),
            node(
                func=clean_traffic_data,
                inputs="parsed_traffic_data",
                outputs="cleaned_traffic_data",
                name="clean_traffic_data_node",
            ),
        ]
    )