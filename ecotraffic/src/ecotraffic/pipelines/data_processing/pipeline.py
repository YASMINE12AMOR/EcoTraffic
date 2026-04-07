from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    fetch_traffic_data_from_mongodb,
    parse_traffic_records,
    clean_traffic_data,
    #add_time_features
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=fetch_traffic_data_from_mongodb,
            inputs={
                "connection_string": "params:mongodb.connection_string",
                "database_name": "params:mongodb.database_name",
                "collection_name": "params:mongodb.collection_name",
                "query": "params:mongodb.query",
                "limit": "params:mongodb.limit"
            },
            outputs="traffic_raw_data",
            name="fetch_mongodb_data"
        ),
        node(
            func=parse_traffic_records,
            inputs="traffic_raw_data",
            outputs="traffic_parsed",
            name="parse_records"
        ),
        node(
            func=clean_traffic_data,
            inputs="traffic_parsed",
            outputs="traffic_cleaned",
            name="clean_data"
        )
    ])