# app/flask_app.py
from pathlib import Path
import os, json, joblib
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from werkzeug.middleware.proxy_fix import ProxyFix

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "models"))

# Load once at startup 
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

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "classes": CLASSES}, 200

@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    # accept either a list of records or {data: [...]}
    records = payload.get("data", payload)
    df = pd.DataFrame(records)
    X = prepare_features(df)
    proba = MODEL.predict_proba(X)
    top1_idx = np.argmax(proba, axis=1)
    top1 = [CLASSES[i] for i in top1_idx]
    resp = []
    for i, row in enumerate(df.to_dict(orient="records")):
        item = {"input": row, "pred_class": top1[i]}
        item["proba"] = {c: float(p) for c, p in zip(CASSES:=CLASSES, proba[i])}
        resp.append(item)
    return jsonify(resp), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081, debug=True)
