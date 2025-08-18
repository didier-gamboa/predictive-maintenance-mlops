import os, sys, yaml
from pathlib import Path
from pprint import pprint

# Resolve repo root once. In Docker we set REPO_ROOT=/opt/airflow/repo
REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parents[2]))

CONFIG_MAP = {
    "data":     "configs/data.yaml",
    "features": "configs/features.yaml",
    "labeling": "configs/labeling.yaml",
    "split":    "configs/split.yaml",
    "train":    "configs/train.yaml",
    "tracking": "configs/tracking.yaml",
}

def _read_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

def load_config() -> dict:
    """
    Load all YAML configs anchored at REPO_ROOT.
    Raises FileNotFoundError listing any missing files.
    """
    cfg, missing = {}, []
    for section, relpath in CONFIG_MAP.items():
        p = (REPO_ROOT / relpath).resolve()
        if not p.exists():
            missing.append((section, str(p)))
            continue
        cfg[section] = _read_yaml(p)

    if missing:
        msg = "Missing config files:\n" + "\n".join(f"  - {sec}: {path}" for sec, path in missing)
        # Use a normal exception; Airflow will render it in task logs
        raise FileNotFoundError(msg)

    return cfg

if __name__ == "__main__":
    print(f"REPO_ROOT -> {REPO_ROOT}")
    config = load_config()
    print("Loaded configuration:")
    pprint(config)
