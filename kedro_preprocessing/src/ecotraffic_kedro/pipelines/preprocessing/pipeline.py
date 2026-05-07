from kedro.pipeline import Pipeline, node, pipeline
from .nodes import extract_env_from_mongo, preprocess_env, save_env_to_postgres


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_env_from_mongo,
                inputs=None,
                outputs="raw_env_df",
                name="extract_env_from_mongo_node",
            ),
            node(
                func=preprocess_env,
                inputs="raw_env_df",
                outputs="preprocessed_env_df",
                name="preprocess_env_node",
            ),
            node(
                func=save_env_to_postgres,
                inputs="preprocessed_env_df",
                outputs=None,
                name="save_env_to_postgres_node",
            ),
        ]
    )