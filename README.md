predictive-maintenance-mlops
==============================

End-to-end MLOps solution for predictive maintenance using the Microsoft Azure dataset. Includes training pipeline, model deployment via FastAPI, Dockerized APIs, and technical documentation.

Project Organization
------------

```
predictive-maintenance-mlops/
├── LICENSE     
├── README.md                  
├── Makefile                     # Makefile with commands like `make data` or `make train`                   
├── configs                      # Config files (models and training hyperparameters)
│   └── model1.yaml              
│
├── data                         
│   ├── external                 # Data from third party sources.
│   ├── interim                  # Intermediate data that has been transformed.
│   ├── processed                # The final, canonical data sets for modeling.
│   └── raw                      # The original, immutable data dump.
│
├── docs                         # Project documentation.
│
├── models                       # Trained and serialized models.
│
├── notebooks                    # Jupyter notebooks.
│
├── references                   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  # Generated graphics and figures to be used in reporting.
│
├── requirements.txt             # The requirements file for reproducing the analysis environment.
└── src                          # Source code for use in this project.
    ├── __init__.py              # Makes src a Python module.
    │
    ├── data                     # Data engineering scripts.
    │   ├── build_features.py    
    │   ├── cleaning.py          
    │   ├── ingestion.py         
    │   ├── labeling.py          
    │   ├── splitting.py         
    │   └── validation.py        
    │
    ├── models                   # ML model engineering (a folder for each model).
    │   └── model1      
    │       ├── dataloader.py    
    │       ├── hyperparameters_tuning.py 
    │       ├── model.py         
    │       ├── predict.py       
    │       ├── preprocessing.py 
    │       └── train.py         
    │
    └── visualization        # Scripts to create exploratory and results oriented visualizations.
        ├── evaluation.py        
        └── exploration.py       
```


--------

## Dataset Setup

This project uses the [Microsoft Azure Predictive Maintenance Dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) available on Kaggle.

To reproduce the dataset locally, follow the steps below.

### 1. Configure Kaggle API Credentials

1. Go to [https://www.kaggle.com](https://www.kaggle.com) and log into your account.
2. Click your profile picture (top right), then select **Settings**.
3. Scroll down to the **API** section and click **"Create New API Token"**.
4. A file named `kaggle.json` will be downloaded automatically.

Open `kaggle.json` and extract your credentials to create a `.env` file at the project root with the following content:

```bash
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

### 2. Download the Dataset

Once your `.env` file is ready, run the following command from the root of the project:

```python
python src/data/download.py
```

This will:

- Load Kaggle credentials from the `.env` file
- Download the dataset using `kagglehub`
- Move the following CSV files to `data/raw/`:

```bash
PdM_telemetry.csv
PdM_errors.csv
PdM_failures.csv
PdM_machines.csv
PdM_maint.csv
```