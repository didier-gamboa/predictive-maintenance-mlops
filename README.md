# Predictive Maintenance MLOps  

End-to-end **MLOps solution** for predictive maintenance using the Microsoft Azure dataset.  
This project implements the full ML lifecycle:  

- **Data ingestion & preprocessing** (ETL pipelines, feature engineering, labeling, train/test split)  
- **Training & experiment tracking** with MLflow  
- **Orchestration** using Apache Airflow  
- **Model deployment** via Flask REST API  
- **Visualization & monitoring** via Streamlit  
- **Containerized services** with Docker Compose  



## 📂 Project Organization  

```
predictive-maintenance-mlops/
├── LICENSE
├── README.md
├── Makefile                     # Utility commands (make train, make test, etc.)
│
├── configs                      # YAML configuration files
│   ├── data.yaml
│   ├── features.yaml
│   ├── labeling.yaml
│   ├── split.yaml
│   ├── tracking.yaml
│   └── train.yaml
│
├── dags                         # Airflow DAGs
│   └── predictive_maintenance_dag.py
│
├── data                         # Data lakehouse folders
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
│
├── docs                         # Documentation
│
├── mlruns                       # MLflow tracking logs
├── models                       # Trained models and artifacts
│
├── notebooks                    # Jupyter notebooks (EDA, feature engineering, modeling)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src                          # Source code
│   ├── common                   # Logging, config & tracking utils
│   │   ├── config.py
│   │   ├── logging.py
│   │   └── tracking.py
│   │
│   ├── data                     # Data pipeline (ingestion, preprocessing, etc.)
│   ├── models                   # Training, prediction & evaluation
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   └── visualization            # Visualization scripts
│
├── app                          # Deployment
│   └── flask_app.py             # Flask REST API
│
├── requirements-base.txt        # Base dependencies
├── requirements-api.txt         # API dependencies
├── requirements-streamlit.txt   # Streamlit dependencies
│
├── docker-compose.airflow.yml   # Multi-service Docker setup
├── dockerfile.airflow
├── dockerfile.api
├── dockerfile.streamlit
└── environment.yml              # Conda environment
```



## ⚙️ Setup & Installation  

### 1. Clone repository  

```bash
git clone https://github.com/<your-username>/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops
```

### 2. Configure Kaggle credentials  

Create a `.env` file in the project root:  

```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### 3. Download dataset  

```bash
python src/data/download.py
```

This will download the **Microsoft Azure Predictive Maintenance Dataset** into `data/raw/`.



## 🐳 Running the Full Stack with Docker  

The system runs fully containerized with **Docker Compose**.  

### Step 1 — Initialize Airflow  

Before first use, run:  

```bash
docker-compose -f docker-compose.airflow.yml up airflow-init
```

This will:  
- Migrate the Airflow DB  
- Create the default **Admin user** (`admin/admin`)  

You should see output confirming user creation.  



### Step 2 — Start Core Services  

Run the core Airflow services:  

```bash
docker-compose -f docker-compose.airflow.yml up -d     airflow-scheduler     airflow-webserver
```

Airflow UI should be available at [http://localhost:8080](http://localhost:8080).  

Login with:  
- **Username**: `admin`  
- **Password**: `admin`  



### Step 3 — Start Optional Services  

#### MLflow  

```bash
docker-compose -f docker-compose.airflow.yml up -d mlflow
```

MLflow UI → [http://localhost:5500](http://localhost:5500)  

#### Flask API  

```bash
docker-compose -f docker-compose.airflow.yml up -d api
```

API → [http://localhost:8081](http://localhost:8081)  

#### Streamlit Dashboard  

```bash
docker-compose -f docker-compose.airflow.yml up -d streamlit
```

Streamlit → [http://localhost:8501](http://localhost:8501)  



## 📊 Workflow  

1. **Data Pipeline**  
   Airflow DAG pulls raw data → processes into features → saves in `/data/processed`.  

2. **Model Training**  
   Trigger training DAG in Airflow or run manually:  

   ```bash
   python src/models/train.py
   ```

   Results are logged to MLflow.  

3. **Experiment Tracking**  
   Compare models in MLflow UI and select the best.  

4. **Deployment**  
   The selected model is exported into `/models` and served via Flask API.    


## ✅ Example API Usage  

### Healthcheck  

```bash
curl http://localhost:8081/health
```

### Predict  

```bash
curl -X POST http://localhost:8081/predict \
  -H "Content-Type: application/json" \
  -d '{
    "volt": 168.2,
    "rotate": 420.5,
    "pressure": 112.9,
    "vibration": 46.3,

    "volt_mean_24h": 167.9,
    "volt_std_24h": 1.8,
    "rotate_mean_24h": 421.2,
    "rotate_std_24h": 3.1,
    "pressure_mean_24h": 113.1,
    "pressure_std_24h": 0.9,
    "vibration_mean_24h": 45.8,
    "vibration_std_24h": 1.2,

    "error1_count_24h": 0,
    "error2_count_24h": 1,
    "error3_count_24h": 0,
    "error4_count_24h": 0,
    "error5_count_24h": 0,

    "comp1": 12.0,
    "comp2": 33.0,
    "comp3": 7.0,
    "comp4": 55.0,

    "model": "M",
    "age": 8
  }'
```

### Response
The API returns per-class probabilities plus the top-1 class:
```
{
  "proba_comp1": 0.04,
  "proba_comp2": 0.07,
  "proba_comp3": 0.12,
  "proba_comp4": 0.10,
  "proba_none": 0.67,
  "pred_class": "none"
}
```

## 🔮 Next Steps  

- Add CI/CD with GitHub Actions.  
- Extend monitoring with Prometheus + Grafana.  