from pathlib import Path
import argparse, json, joblib
import pandas as pd
import numpy as np

def batch_predict(input_csv: str, output_csv: str, artifacts_dir: str = "models") -> str:
    model = joblib.load(Path(artifacts_dir) / "champion.pkl")
    classes = json.loads((Path(artifacts_dir) / "classes.json").read_text())["classes"]

    df = pd.read_csv(input_csv)
    # Drop labels if accidentally present
    for col in ["failure_comp", "failure_binary"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    proba = model.predict_proba(df)
    top1_idx = np.argmax(proba, axis=1)
    top1 = [classes[i] for i in top1_idx]
    proba_df = pd.DataFrame(proba, columns=[f"proba_{c}" for c in classes])

    out = pd.concat([df.reset_index(drop=True), proba_df, pd.Series(top1, name="pred_class")], axis=1)
    out.to_csv(output_csv, index=False)
    return output_csv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--artifacts_dir", default="models")
    args = ap.parse_args()
    print(batch_predict(args.input, args.output, args.artifacts_dir))
