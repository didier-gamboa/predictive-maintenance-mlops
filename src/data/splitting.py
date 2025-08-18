from pathlib import Path
import pandas as pd
from src.common.logging import get_logger

log = get_logger("splitting")

def time_fraction_with_gap(df: pd.DataFrame, time_col: str, train_fraction: float, gap_hours: int):
    all_times = df[time_col].dropna().sort_values().unique()
    cut_idx = int(len(all_times) * train_fraction)
    cut_idx = min(max(cut_idx, 1), len(all_times) - 2)
    cut_time = pd.to_datetime(all_times[cut_idx])
    gap = pd.Timedelta(hours=gap_hours)
    last_train, first_test = cut_time - gap, cut_time
    train = df[df[time_col] <  last_train]
    test  = df[df[time_col] >  first_test]
    return train, test, last_train, first_test

def run(cfg_data: dict, cfg_split: dict) -> dict:
    proc = Path(cfg_data["processed_dir"])
    time_col = cfg_data["keys"]["time"]
    df = pd.read_csv(proc / "dataset.csv", parse_dates=[time_col])

    train, test, last_train, first_test = time_fraction_with_gap(
        df, time_col, cfg_split["time_fraction"]["train_fraction"], cfg_split["time_fraction"]["gap_hours"]
    )

    train_p = proc / "train.csv"; test_p = proc / "test.csv"
    train.to_csv(train_p, index=False); test.to_csv(test_p, index=False)
    log.info(f"Split: train<{last_train}  GAP[{last_train},{first_test}]  test>{first_test}")
    return {"train_csv": str(train_p), "test_csv": str(test_p)}

if __name__ == "__main__":
    from src.common.config import load_config
    cfg = load_config()
    run(cfg["data"], cfg["split"])
