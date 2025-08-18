from pathlib import Path
import pandas as pd
from src.common.logging import get_logger

log = get_logger("labeling")

def build_labels(features_df: pd.DataFrame, failures: pd.DataFrame,
                 id_col: str, time_col: str, horizon_h: int,
                 none_class: str, allowed: list[str]) -> pd.DataFrame:
    labels = features_df[[id_col, time_col]].copy()
    labels["failure_binary"] = 0
    labels["failure_comp"]   = none_class

    fail = failures.rename(columns={time_col: "fail_datetime"}).sort_values([id_col, "fail_datetime"])

    for mid, grp in fail.groupby(id_col):
        m = labels[id_col] == mid
        lab_m = labels.loc[m, [time_col, "failure_binary", "failure_comp"]].copy()
        for _, ev in grp.iterrows():
            f_time = ev["fail_datetime"]; comp = str(ev["failure"])
            if allowed and comp not in allowed:
                continue
            start = f_time - pd.Timedelta(hours=horizon_h)
            mask = (lab_m[time_col] >= start) & (lab_m[time_col] < f_time)
            lab_m.loc[mask, "failure_binary"] = 1
            lab_m.loc[mask, "failure_comp"]   = comp
        labels.loc[m, ["failure_binary","failure_comp"]] = lab_m[["failure_binary","failure_comp"]].values
    return labels

def run(cfg_data: dict, cfg_feat: dict, cfg_lab: dict) -> dict:
    proc = Path(cfg_data["processed_dir"])
    time_col = cfg_data["keys"]["time"]; id_col = cfg_data["keys"]["id"]
    allowed = cfg_lab.get("allowed_failure_classes", ["comp1","comp2","comp3","comp4"])
    none_class = cfg_lab.get("none_class", "none")
    horizon = int(cfg_lab["horizon_hours"])

    feats = pd.read_csv(proc / "features.csv", parse_dates=[time_col])
    failures = pd.read_csv(proc / "failures_clean.csv", parse_dates=[time_col])

    labels = build_labels(feats, failures, id_col, time_col, horizon, none_class, allowed)
    dataset = feats.merge(labels, on=[id_col, time_col], how="left")

    out_csv = proc / "dataset.csv"
    dataset.to_csv(out_csv, index=False)
    log.info(f"Dataset -> {out_csv} rows={len(dataset)}")
    return {"dataset_csv": str(out_csv)}

if __name__ == "__main__":
    from src.common.config import load_config
    cfg = load_config()
    run(cfg["data"], cfg["features"], cfg["labeling"])