# Technical Design Document — Predictive Maintenance MLOps

The scope of this document is to describe the architecture, data flow, core components, interfaces, and key decisions for the end-to-end machine learning system that predicts component failures.

## 1. Problem Overview & Goals

The main business problem is to reduce unplanned downtime by predicting whether a machine will fail—and if so, which component is at risk—using historical telemetry, error counts, and machine metadata.

In this case, we consider a multiclass classification with 5 outcomes: comp1, comp2, comp3, comp4, or none (no imminent failure). The API also surfaces calibrated per-class probabilities.

Success criteria (initial):
- Model: macro-F1 ≥ baseline; Top-1 accuracy ≥ baseline; AUROC (one-vs-rest) reported per class.
- Ops: fully containerized stack that runs with docker compose; jobs orchestrated in Airflow; experiments tracked in MLflow; model served via Flask API; optional Streamlit dashboard.
- Data: leakage-safe, time-aware split; reproducible feature generation.


## 2. High-Level Architecture

- **Data Ingestion**  
  - Download and store raw data from the source (e.g., Kaggle/Azure dataset).  
  - Validate schema and datatypes.  
  - Produce intermediate datasets for downstream processing.  

- **Orchestration — Airflow**  
  - Defines a **DAG** with chained tasks: ingestion, feature engineering, training, evaluation.  
  - Ensures reproducible, scheduled, and monitored execution.  

- **Experiment Tracking — MLflow**  
  - Logs parameters, metrics, and artifacts for each training run.  
  - Enables experiment comparison to identify the best-performing model.   

- **Exported Artifacts**  
  - Serialized models that include the preprocessing pipeline.  
  - Stored under `/models/` for consumption by the serving layer.  
- **Model Serving — Flask REST API**  
  - Provides `/health` endpoint for liveness checks.  
  - Provides `/predict` endpoint for real-time inference.  
  - Automatically loads the latest exported production-ready model.  

- **Infrastructure — Docker Compose**  
  - Orchestrates all services (Airflow, MLflow, API, Streamlit) into a single reproducible stack.  
  - Exposes UIs on fixed ports: Airflow (8080), MLflow (5500), API (8081), Streamlit (8501).   

## 3. Repository & Runtime Layout

```
/app/
  flask_app.py                # REST API (/health, /predict)
/configs/
  data.yaml                   # paths & dataset settings
  features.yaml               # features & rolling windows
  labeling.yaml               # target construction rules
  split.yaml                  # time split & gap settings
  train.yaml                  # model grids & training settings
  tracking.yaml               # MLflow URI, experiment name
/dags/
  predictive_maintenance_dag.py
/mlruns/                      # MLflow local store
/src/
  common/                     # config, logging, tracking utils
  data/                       # download, clean, feature build, label
  models/                     # train, evaluate, predict, export
  visualization/              # EDA/eval plots
/models/                      # exported production artifacts
docker-compose.airflow.yml    # local stack (Airflow, MLflow, API, Streamlit)
dockerfile.airflow
dockerfile.api
dockerfile.streamlit
README.md
```

**Default ports:** Airflow `8080`, MLflow `5500`, API `8081`, Streamlit `8501`.



## 4. Data Model & Features

**Core fields (inference schema):**
- **Telemetry:** `volt`, `rotate`, `pressure`, `vibration`
- **Rolling 24h stats:** `volt_mean_24h`, `volt_std_24h`, `rotate_mean_24h`, `rotate_std_24h`, `pressure_mean_24h`, `pressure_std_24h`, `vibration_mean_24h`, `vibration_std_24h`
- **Error counts (24h):** `error1_count_24h` ... `error5_count_24h`
- **Maintenance features:** `comp1`, `comp2`, `comp3`, `comp4`
- **Metadata:** `model` (categorical), `age` (numeric)

**Targets:**
- Multiclass label `y ∈ {comp1, comp2, comp3, comp4, none}`.
- Optional binary `y_bin = 1{y ≠ none}` for monitoring/calibration.


## 5. Data Flow & Split Strategy

1. **Ingestion & validation**
   - Download raw data to `data/raw/` (e.g., via Kaggle/HTTP).
   - Validate required columns and dtypes, enforce monotonic time per `machineID`.

2. **Feature engineering**
   - Compute rolling means/std (24h) for telemetry.
   - Compute 24h error counts per error code.
   - Encode categorical `model` and build maintenance features.

3. **Labeling**
   - Define failure windows; assign first component failure ahead of timestamp, else `none`.
   - Optionally define prediction horizon `H` hours.

4. **Time-aware split & leakage guard**
   - Train/test split by timestamp.
   - Enforce **gap** between `last_train` and `first_test`: set `gap ≥ max_rolling_window - 1` (with 24h windows → gap ≈ 24h).
   - This prevents sharing rolling contexts across train/test windows.

## 6. Modeling & Evaluation

**Algorithms:**
- `HistGradientBoostingClassifier` (scikit-learn)
- `XGBoost` (xgboost)

**Hyperparameters (compact grids):**
- HGB: `max_depth`, `learning_rate`, `min_samples_leaf`, `l2_regularization`
- XGB: `max_depth`, `eta`, `subsample`, `colsample_bytree`, `lambda`, `gamma`

**Metrics:**
- Primary: Macro-F1, Top-1 Accuracy


## 7 Orchestration — Airflow

The project uses **Apache Airflow** to orchestrate the end-to-end machine learning workflow.  
The DAG `predictive_maintenance_etl_train_eval` defines a clear pipeline with sequential tasks:

- **Ingest (`ingest`)**  
  - Loads raw data from the source.  
  - Stores it in the local `data/` directory after schema validation.  

- **Feature Engineering (`feature_engineering`)**  
  - Builds rolling statistics and error count features.  
  - Prepares enriched datasets for labeling.  

- **Labeling (`labeling`)**  
  - Creates target variables based on machine failure events.  
  - Assigns multiclass labels (`comp1..4`, `none`).  

- **Splitting (`splitting`)**  
  - Performs a **time-aware split** into train/test sets.  
  - Enforces a configurable **gap** window to prevent data leakage from rolling features.  

- **Training (`train`)**  
  - Trains candidate models (HistGradientBoosting, XGBoost) using configs from `/configs/train.yaml`.  
  - Logs parameters, metrics, and artifacts to **MLflow**.  

- **Evaluation (`evaluate`)**  
  - Computes metrics such as Macro-F1, Accuracy, AUROC.  
  - Logs evaluation reports and plots to MLflow.  
  - Prepares the best model for export.  

**DAG Configuration Highlights:**
- **Owner:** `mlops`  
- **Retries:** 1 (with 2-minute delay)  
- **Trigger:** Manual (`schedule_interval=None`) — can be changed to cron for automation.  
- **Tags:** `["pm", "mlops", "mlflow"]` for easier filtering in Airflow UI.  

**Execution Flow:**  
`ingest → feature_engineering → labeling → splitting → train → evaluate`  

This orchestration ensures that every stage — from raw data ingestion to model evaluation — is reproducible, traceable, and integrated with the tracking system (MLflow).  


## 8. Experiment Tracking (MLflow)

- `tracking.yaml` defines `tracking_uri` and `experiment`.
- Log params, metrics, figures (confusion matrix, ROC-OvR), and serialized model (with preprocessing).


## 9. Serving API (Flask)

**Endpoints:**
- `GET /health` → `{"status": "ok"}`
- `POST /predict` → returns per-class probabilities and top-1 label

**Request example:**
```json
{
  "volt": 168.2, "rotate": 420.5, "pressure": 112.9, "vibration": 46.3,
  "volt_mean_24h": 167.9, "volt_std_24h": 1.8, "rotate_mean_24h": 421.2, "rotate_std_24h": 3.1,
  "pressure_mean_24h": 113.1, "pressure_std_24h": 0.9, "vibration_mean_24h": 45.8, "vibration_std_24h": 1.2,
  "error1_count_24h": 0, "error2_count_24h": 1, "error3_count_24h": 0, "error4_count_24h": 0, "error5_count_24h": 0,
  "comp1": 12.0, "comp2": 33.0, "comp3": 7.0, "comp4": 55.0,
  "model": "M", "age": 8
}
```

**Response example:**
```json
{
  "proba_comp1": 0.04, "proba_comp2": 0.07, "proba_comp3": 0.12, "proba_comp4": 0.10, "proba_none": 0.67,
  "pred_class": "none"
}
```

The API is under construction. The main source code is provide on `./app/flask_app.py` but it needs to be correctly deployed and debugged. 

**Port:** `8081`. API loads the latest exported model artifact and embedded preprocessing pipeline.



## 10. Dashboard (Streamlit)

The Streamlit Dashboard is under construction. The main source code is provide on `./app/streamlit_app.py` but it needs to be correctly deployed and debugged. 

## 11. Containerization (Docker Compose)

**Key services:** Airflow (webserver/scheduler/worker + Postgres/Redis), MLflow, API, Streamlit.

**Common commands:**
```bash
# 1) Initialize Airflow metadata DB & accounts
docker compose -f docker-compose.airflow.yml up airflow-init

# 2) Start core services (webserver, scheduler, worker, db, redis)
docker compose -f docker-compose.airflow.yml up -d   airflow-webserver airflow-scheduler airflow-worker postgres redis

# 3) Optional UIs/services
docker compose -f docker-compose.airflow.yml up -d mlflow api streamlit
```

## 12. CI/CD (Roadmap)

For **Continuous Integration (CI)**, the project can rely on GitHub Actions to automatically run checks on every pull request and push to the main branch. The pipeline would include code quality verification (using tools like `ruff` for linting and `mypy` for type checking), unit and integration tests with `pytest`, and lightweight security scans for dependencies. Additionally, smoke tests can be run against the Dockerized API service to ensure endpoints like `/health` and `/predict` behave correctly. This guarantees that any change merged into the repository maintains coding standards, passes tests, and keeps the API stable.

For **Continuous Delivery (CD)**, the system can build and push Docker images for the API, Airflow, and Streamlit services after a successful CI run. Images can be versioned with commit SHAs and tags, and scanned for vulnerabilities before release. A model promotion workflow can be added using MLflow’s Model Registry, where models are automatically registered in *Staging* if evaluation metrics meet a baseline, and then promoted to *Production* after manual approval. This setup ensures that both infrastructure and machine learning models are deployed in a controlled, reproducible, and secure manner.


## 13. Observability & Model Health (Roadmap)

Application-level observability can be achieved by monitoring throughput, latency, and error rates of the Flask API, as well as Airflow DAG execution times and SLAs. Standard container metrics such as CPU and memory usage should be collected from each service, ideally through Prometheus and visualized in Grafana dashboards. Logs can be centralized for troubleshooting, making it easier to detect failures in the pipeline or sudden spikes in resource consumption. This layer ensures that the overall system is reliable and operational issues are quickly visible to the engineering team.

For **model health monitoring**, the focus is on ensuring predictions remain accurate and relevant after deployment. Techniques such as schema validation and feature drift detection (e.g., Population Stability Index or Jensen–Shannon divergence) can be scheduled to run periodically against incoming data. Performance backtesting should be conducted when ground truth labels are available, allowing detection of concept drift over time. These practices enable proactive retraining or rollback of models when their quality degrades, protecting business outcomes and user trust in the predictions.
