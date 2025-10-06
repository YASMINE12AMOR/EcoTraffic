from .pipeline import create_pipeline

def register_pipelines():
    return {
        "weather_pollution": create_pipeline(),
        "__default__": create_pipeline(),
    }
