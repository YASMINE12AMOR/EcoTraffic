from ecotraffic_kedro.pipelines.preprocessing.pipeline import create_pipeline


def register_pipelines():
    preprocessing_pipeline = create_pipeline()
    return {
        "__default__": preprocessing_pipeline,
        "preprocessing": preprocessing_pipeline,
    }