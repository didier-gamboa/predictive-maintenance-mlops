from contextlib import contextmanager
import mlflow

def init_mlflow(uri: str, experiment: str):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

@contextmanager
def start_run(run_name: str, tags: dict | None = None):
    with mlflow.start_run(run_name=run_name) as _:
        if tags:
            mlflow.set_tags(tags)
        yield
