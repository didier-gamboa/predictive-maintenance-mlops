from datetime import datetime, timedelta
import os, sys
from pathlib import Path

# Make your repo importable inside the container
REPO_ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1]))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.common.config import load_config
from src.data.ingestion import run as ingest_run
from src.data.feature_engineering import run as feat_run
from src.data.labeling import run as label_run
from src.data.splitting import run as split_run
from src.models.train import run as train_run
from src.models.evaluate import run as eval_run

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="predictive_maintenance_etl_train_eval",
    description="End-to-end: ingest → features → labels → split → train → evaluate",
    start_date=datetime(2025, 8, 1),
    schedule_interval=None,   # manual trigger; change to cron like "0 3 * * *" if desired
    catchup=False,
    default_args=default_args,
    tags=["pm", "mlops", "mlflow"],
) as dag:

    def task_ingest():
        cfg = load_config()
        return ingest_run(cfg["data"])

    def task_features():
        cfg = load_config()
        return feat_run(cfg["data"], cfg["features"])

    def task_label():
        cfg = load_config()
        return label_run(cfg["data"], cfg["features"], cfg["labeling"])

    def task_split():
        cfg = load_config()
        return split_run(cfg["data"], cfg["split"])

    def task_train():
        cfg = load_config()
        return train_run(cfg["data"], cfg["train"], cfg["tracking"])

    def task_evaluate():
        cfg = load_config()
        return eval_run(cfg["data"], cfg["train"], cfg["tracking"])

    ingest = PythonOperator(task_id="ingest", python_callable=task_ingest)
    features = PythonOperator(task_id="feature_engineering", python_callable=task_features)
    labeling = PythonOperator(task_id="labeling", python_callable=task_label)
    splitting = PythonOperator(task_id="splitting", python_callable=task_split)
    training = PythonOperator(task_id="train", python_callable=task_train)
    evaluating = PythonOperator(task_id="evaluate", python_callable=task_evaluate)

    ingest >> features >> labeling >> splitting >> training >> evaluating
