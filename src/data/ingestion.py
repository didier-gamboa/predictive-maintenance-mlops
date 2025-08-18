from pathlib import Path
import pandas as pd
from src.common.logging import get_logger

log = get_logger("Ingestion")

def run(cfg_data: dict) -> dict:
    raw = Path(cfg_data["raw_dir"])
    proc = Path(cfg_data["processed_dir"]); proc.mkdir(parents=True, exist_ok=True)
    f = cfg_data["files"]; time_col = cfg_data["keys"]["time"]; id_col = cfg_data["keys"]["id"]

    # Load
    telemetry = pd.read_csv(raw / f["telemetry"], parse_dates=[time_col])
    errors    = pd.read_csv(raw / f["errors"],    parse_dates=[time_col])
    maint     = pd.read_csv(raw / f["maintenance"], parse_dates=[time_col])
    failures  = pd.read_csv(raw / f["failures"],  parse_dates=[time_col])
    machines  = pd.read_csv(raw / f["machines"])

    # Basic checks
    for df, name in [(telemetry,"telemetry"),(errors,"errors"),(maint,"maintenance"),(failures,"failures")]:
        assert time_col in df.columns, f"{name}: missing timestamp col {time_col}"
        assert id_col in df.columns,   f"{name}: missing id col {id_col}"

    # Sort
    telemetry = telemetry.sort_values([id_col, time_col])
    errors    = errors.sort_values([id_col, time_col])
    maint     = maint.sort_values([id_col, time_col])
    failures  = failures.sort_values([id_col, time_col])

    # Save CSVs
    paths = {
        "telemetry":   str(proc / "telemetry_clean.csv"),
        "errors":      str(proc / "errors_clean.csv"),
        "maintenance": str(proc / "maintenance_clean.csv"),
        "failures":    str(proc / "failures_clean.csv"),
        "machines":    str(proc / "machines_clean.csv"),
    }
    telemetry.to_csv(paths["telemetry"],   index=False)
    errors.to_csv(paths["errors"],         index=False)
    maint.to_csv(paths["maintenance"],     index=False)
    failures.to_csv(paths["failures"],     index=False)
    machines.to_csv(paths["machines"],     index=False)

    log.info(f"Clean CSVs written to {proc}")
    return paths

if __name__ == "__main__":
    from src.common.config import load_config
    cfg = load_config()
    run(cfg["data"])
