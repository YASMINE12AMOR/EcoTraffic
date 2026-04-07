from kedro.pipeline import Pipeline
from ecotraffic.pipelines.data_processing.pipeline import create_pipeline as data_processing


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "data_processing": data_processing(),
        "__default__": data_processing(),  # optionnel : lance ce pipeline si tu fais "kedro run" sans --pipeline
    }
