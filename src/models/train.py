# src/models/train.py
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from src.common.logging import get_logger
from src.common.tracking import init_mlflow, start_run

log = get_logger("train")


def _build_preprocessor(cat_cols: list[str], num_cols: list[str], one_hot_cap: int):
    transformers = []
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", max_categories=one_hot_cap, sparse_output=False),
            cat_cols
        ))

    return ColumnTransformer(
        transformers=transformers + [("num", "passthrough", num_cols)],
        remainder="drop"
    )


def _build_estimator(name: str, num_classes: int, params: dict):
    if name == "hgb":
        return HistGradientBoostingClassifier(loss="log_loss", random_state=42, **params)
    if name == "xgboost":
        return XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            **params
        )
    raise ValueError(f"Unknown estimator {name}")


def run(cfg_data: dict, cfg_train: dict, cfg_track: dict) -> dict:
    # MLflow
    Path(cfg_track["mlflow_uri"]).mkdir(parents=True, exist_ok=True)
    init_mlflow(cfg_track["mlflow_uri"], cfg_track["experiment"])

    proc = Path(cfg_data["processed_dir"])
    df_train = pd.read_csv(proc / "train.csv", parse_dates=[cfg_data["keys"]["time"]])

    time_col = cfg_data["keys"]["time"]
    id_col   = cfg_data["keys"]["id"]

    y_col = cfg_train["target_mc"]
    # drop targets + keys from X
    drop_cols = [c for c in [y_col, cfg_train.get("target_bin"), time_col, id_col] if c in df_train.columns]
    X = df_train.drop(columns=drop_cols)
    y = df_train[y_col].astype(str)

    # decide categorical + numeric explicitly
    cat_cols = ["model"] if "model" in X.columns else []
    # drop any stray non-numeric columns except 'model'
    non_numeric = [c for c in X.columns if c not in cat_cols and not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        log.info(f"Dropping non-numeric columns (not in cat_cols): {non_numeric}")
        X = X.drop(columns=non_numeric)
    num_cols = [c for c in X.columns if c not in cat_cols]

    # label encoding for search/estimators
    classes = sorted(pd.Index(pd.unique(y)).tolist())
    num_classes = len(classes)
    label2id = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([label2id[v] for v in y], dtype=int)

    # build preprocessor with explicit columns
    pre = _build_preprocessor(cat_cols, num_cols, cfg_train.get("one_hot_max_categories", 16))

    def make_pipeline(est_name: str, params: dict):
        est = _build_estimator(est_name, num_classes, params)
        return Pipeline([("prep", pre), ("clf", est)])

    # CV strategy for search
    cv_cfg = cfg_train["search"]["cv"]
    if cv_cfg["type"] == "timeseries":
        cv = TimeSeriesSplit(n_splits=cv_cfg["n_splits"])
    else:
        cv = StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=True, random_state=cfg_train["seed"])

    with start_run(run_name=f"{cfg_track['run_name_prefix']}_search", tags=cfg_track.get("tags", {})):
        mlflow.log_param("classes", classes)
        mlflow.log_param("search_estimators", cfg_train["search"]["estimators"])
        mlflow.log_param("search_strategy", cfg_train["search"]["strategy"])

        best_model, best_score, best_name = None, -np.inf, None

        for est_name in cfg_train["search"]["estimators"]:
            grid = cfg_train["search"]["grids"][est_name]
            pipe = make_pipeline(est_name, params={})

            if cfg_train["search"]["strategy"] == "grid":
                searcher = GridSearchCV(
                    pipe,
                    param_grid={f"clf__{k}": v for k, v in grid.items()},
                    scoring=cfg_train["search"]["scoring"],
                    cv=cv,
                    n_jobs=-1,
                    verbose=1
                )
            else:
                searcher = RandomizedSearchCV(
                    pipe,
                    param_distributions={f"clf__{k}": v for k, v in grid.items()},
                    n_iter=cfg_train["search"]["n_iter"],
                    scoring=cfg_train["search"]["scoring"],
                    cv=cv,
                    n_jobs=-1,
                    verbose=1,
                    random_state=cfg_train["seed"]
                )

            searcher.fit(X, y_enc)
            cv_score = float(searcher.best_score_)
            mlflow.log_metric(f"{est_name}_cv_{cfg_train['search']['scoring']}", cv_score)
            mlflow.log_params({f"{est_name}_best_{k}": v for k, v in searcher.best_params_.items()})

            if cv_score > best_score:
                best_score = cv_score
                best_model = searcher.best_estimator_
                best_name  = est_name

        # Persist artifacts
        artifacts = Path(cfg_track["artifacts_dir"]); artifacts.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, artifacts / "champion.pkl")
        (artifacts / "classes.json").write_text(json.dumps({"classes": classes}, indent=2))

        # Log to MLflow
        mlflow.sklearn.log_model(best_model, artifact_path="model")
        mlflow.log_artifact(artifacts / "classes.json")
        mlflow.log_param("final_model_name", best_name)
        mlflow.log_metric("cv_best_score", best_score)

        log.info(f"Best model: {best_name}  CV_{cfg_train['search']['scoring']}={best_score:.4f}")
        return {"model_path": str(artifacts / "champion.pkl"), "classes": classes}


if __name__ == "__main__":
    from src.common.config import load_config
    cfg = load_config()
    run(cfg["data"], cfg["train"], cfg["tracking"])