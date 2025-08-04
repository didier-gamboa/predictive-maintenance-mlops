import os
from pathlib import Path
from dotenv import load_dotenv
from kagglehub import dataset_download

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=env_path)

RAW_DATA_PATH = Path("data/raw/")
EXPECTED_FILES = [
    "PdM_telemetry.csv",
    "PdM_errors.csv",
    "PdM_failures.csv",
    "PdM_machines.csv",
    "PdM_maint.csv",
]

def check_credentials():
    '''Verifies if Kaggle credentials are available.'''
    return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))

def data_already_downloaded():
    '''Verifies if all expected files already exists.'''
    return all((RAW_DATA_PATH / file).exists() for file in EXPECTED_FILES)

def download_dataset():
    '''Downloads and stores the Kaggle dataset into data/raw/.'''
    if data_already_downloaded():
        print("Dataset already exists in data/raw/")
        return RAW_DATA_PATH
    if not check_credentials():
        raise RuntimeError("Kaggle credentials not found. Please set them in a .env file or as environment variables.")

    print("Downloading dataset from Kaggle...")
    dataset_dir = dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")

    for file in Path(dataset_dir).glob("*.csv"):
        destination = RAW_DATA_PATH / file.name
        file.replace(destination)

    print("All dataset files downloaded and stored in data/raw/")
    return RAW_DATA_PATH

if __name__ == "__main__":
    download_dataset()