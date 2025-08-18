# src/models/evaluate.py
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from src.common.logging import get_logger

log = get_logger("evaluate")

KEY_COLS = ["datetime", "machineID"]
LABEL_COLS = ["failure_comp", "failure_binary"]

def failure_only_macro_f1(y_true, y_pred):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    mask = y_true != "none"
    if mask.sum() == 0:
        return 0.0
    return f1_score(y_true[mask], y_pred[mask], average="macro")

def _prepare_X_for_eval(df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    # Quitar columnas que no son features
    drop_cols = [c for c in (LABEL_COLS + KEY_COLS + [y_col]) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Mantener numéricas + 'model' 
    cat_cols = ["model"] if "model" in df.columns else []
    num_cols = [c for c in df.columns if c not in cat_cols]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "model" in df.columns:
        df["model"] = df["model"].astype("string").fillna("unknown")

    keep = (["model"] if "model" in df.columns else []) + num_cols
    return df[keep].copy()

def run(cfg_data: dict, cfg_train: dict, cfg_track: dict) -> dict:
    proc = Path(cfg_data["processed_dir"])
    time_col = cfg_data["keys"]["time"]

    df_test = pd.read_csv(proc / "test.csv", parse_dates=[time_col])

    y_col = cfg_train["target_mc"]
    y_true = df_test[y_col].astype(str)

    X = _prepare_X_for_eval(df_test.copy(), y_col=y_col)

    # Cargar modelo y clases
    artifacts_dir = Path(cfg_track["artifacts_dir"])
    model = joblib.load(artifacts_dir / "champion.pkl")
    classes = json.loads((artifacts_dir / "classes.json").read_text())["classes"]

    y_idx = model.predict(X)
    y_pred = pd.Series([classes[i] for i in y_idx], index=y_true.index)

    # Métricas
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro_fail_only": float(failure_only_macro_f1(y_true, y_pred)),
        "labels": classes,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, target_names=classes, zero_division=0, output_dict=True
        ),
    }

    out = artifacts_dir / "evaluation.json"
    out.write_text(json.dumps(metrics, indent=2))
    log.info(f"Evaluation -> {out}")
    return {"evaluation": str(out)}

if __name__ == "__main__":
    from src.common.config import load_config
    cfg = load_config()
    run(cfg["data"], cfg["train"], cfg["tracking"])