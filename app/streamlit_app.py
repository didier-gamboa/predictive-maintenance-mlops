# app/streamlit_app.py
from pathlib import Path
import os, json, joblib
import streamlit as st
import pandas as pd
import numpy as np

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "models"))
MODEL = joblib.load(ARTIFACTS_DIR / "champion.pkl")
CLASSES = json.loads((ARTIFACTS_DIR / "classes.json").read_text())["classes"]

KEY_COLS = ["datetime", "machineID"]
LABEL_COLS = ["failure_comp", "failure_binary"]

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in (LABEL_COLS + KEY_COLS) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    cat_cols = ["model"] if "model" in df.columns else []
    num_cols = [c for c in df.columns if c not in cat_cols]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "model" in df.columns:
        df["model"] = df["model"].astype("string").fillna("unknown")
    keep = (["model"] if "model" in df.columns else []) + num_cols
    return df[keep].copy()

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance — Multiclass Failure Prediction")

st.sidebar.header("Upload CSV")
file = st.sidebar.file_uploader("Upload telemetry features CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    st.subheader("Preview")
    st.dataframe(df.head())

    X = prepare_features(df)
    proba = MODEL.predict_proba(X)
    top1_idx = np.argmax(proba, axis=1)
    top1 = [CLASSES[i] for i in top1_idx]
    proba_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in CLASSES])
    out = pd.concat([df.reset_index(drop=True), proba_df, pd.Series(top1, name="pred_class")], axis=1)

    st.subheader("Predictions")
    st.dataframe(out.head(200))
    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
else:
    st.info("Upload a CSV to get predictions.")
