from pathlib import Path
import pandas as pd
import numpy as np
from src.common.logging import get_logger

log = get_logger("feature_engineering")

def _build_hourly_grid(telemetry: pd.DataFrame, machines: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
    tmin, tmax = telemetry[time_col].min(), telemetry[time_col].max()
    hourly_index = pd.date_range(start=tmin, end=tmax, freq="1h")
    machines_list = machines[id_col].unique()
    grid = pd.MultiIndex.from_product([machines_list, hourly_index], names=[id_col, time_col]).to_frame(index=False)
    return grid

def _telemetry_rolling(df: pd.DataFrame, sensors: list[str], id_col: str, time_col: str, window: int) -> pd.DataFrame:
    df = df.sort_values([id_col, time_col]).copy()
    g = df.groupby(id_col, group_keys=False)
    for col in sensors:
        if col not in df.columns:
            continue
        df[f"{col}_mean_{window}h"] = g[col].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"{col}_std_{window}h"]  = g[col].transform(lambda s: s.rolling(window, min_periods=1).std())
    return df

def _error_counts(errors: pd.DataFrame, grid: pd.DataFrame, id_col: str, time_col: str, error_col: str, window: int) -> pd.DataFrame:
    # One-hot per hour then rolling sums over window
    err_pivot = (errors.assign(flag=1)
                       .pivot_table(index=[id_col, time_col], columns=error_col, values="flag", fill_value=0)
                       .reset_index())
    df = grid.merge(err_pivot, on=[id_col, time_col], how="left")
    err_cols = [c for c in df.columns if c.startswith("error")]
    if err_cols:
        df[err_cols] = df[err_cols].fillna(0)
        df = df.sort_values([id_col, time_col])
        g = df.groupby(id_col, group_keys=False)
        for c in err_cols:
            df[f"{c}_count_{window}h"] = g[c].transform(lambda s: s.rolling(window, min_periods=1).sum())
    keep = [id_col, time_col] + [c for c in df.columns if c.endswith(f"_count_{window}h")]
    return df[keep]

def _maint_days_since(maint: pd.DataFrame, grid: pd.DataFrame, id_col: str, time_col: str, comp_col: str) -> pd.DataFrame:
    # One-hot per component, group hourly, merge on grid, forward-fill last replacement timestamp, compute days since
    oh = pd.get_dummies(maint, columns=[comp_col]).rename(columns=lambda c: c.replace(f"{comp_col}_", ""))
    comp_cols = [c for c in oh.columns if c.startswith("comp")]
    if not comp_cols:
        return grid[[id_col, time_col]].copy()
    hourly = oh.groupby([id_col, time_col])[comp_cols].sum().reset_index()
    df = grid.merge(hourly, on=[id_col, time_col], how="left").sort_values([id_col, time_col])
    df[comp_cols] = df[comp_cols].fillna(0)
    out = df[[id_col, time_col]].copy()
    for c in comp_cols:
        evt_ts = df[time_col].where(df[c] >= 1, pd.NaT)
        last_rep = df.assign(_evt=evt_ts).groupby(id_col)["_evt"].ffill()
        out[c] = (df[time_col] - last_rep).dt.total_seconds() / 86400.0
    return out

def run(cfg_data: dict, cfg_feat: dict) -> dict:
    proc = Path(cfg_data["processed_dir"])
    time_col = cfg_data["keys"]["time"]; id_col = cfg_data["keys"]["id"]
    sensors  = cfg_data.get("sensors", ["volt","rotate","pressure","vibration"])
    error_col = cfg_data.get("errors_col", "errorID")
    comp_col  = cfg_data.get("maint_col", "comp")

    # Read clean CSVs with parsed timestamps
    telemetry = pd.read_csv(proc / "telemetry_clean.csv",   parse_dates=[time_col])
    errors    = pd.read_csv(proc / "errors_clean.csv",      parse_dates=[time_col])
    maint     = pd.read_csv(proc / "maintenance_clean.csv", parse_dates=[time_col])
    machines  = pd.read_csv(proc / "machines_clean.csv")

    # Grid
    grid = _build_hourly_grid(telemetry, machines, id_col, time_col)

    # Telemetry on grid + rolling (24h)
    win = int(cfg_feat["windows_hours"][0])  # = 24 from your yaml
    df = grid.merge(telemetry[[id_col, time_col] + sensors], on=[id_col, time_col], how="left")
    df = _telemetry_rolling(df, sensors, id_col, time_col, window=win)

    # Errors with 24h counts
    if cfg_feat.get("use_error_counts", True):
        errf = _error_counts(errors, grid, id_col, time_col, error_col, window=win)
        df = df.merge(errf, on=[id_col, time_col], how="left")

    # Maintenance with days since replacement
    if cfg_feat.get("maint_time_since_replacement_days", True):
        maintf = _maint_days_since(maint, grid, id_col, time_col, comp_col)
        df = df.merge(maintf, on=[id_col, time_col], how="left")

    # Machine meta
    if cfg_feat.get("use_machine_meta", True):
        df = df.merge(machines[[id_col, "model", "age"]], on=id_col, how="left")

    out_csv = proc / "features.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"Features -> {out_csv} rows={len(df)} cols={len(df.columns)}")
    return {"features_csv": str(out_csv)}

if __name__ == "__main__":
    from src.common.config import load_config
    cfg = load_config()
    run(cfg["data"], cfg["features"])
