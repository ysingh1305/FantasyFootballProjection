"""
Controlled experiments for the fantasy football final project.

This script adds the paper-alignment experiments that are awkward to maintain
inside notebooks:

1. Regression feature ablations:
   - rolling_only
   - rolling_plus_previous_season
   - full_features, including opponent defensive rolling features

2. Regression model variants:
   - linear_regression baseline
   - lightgbm_expected
   - lightgbm_boom_weighted, which upweights high-scoring games to reduce
     systematic underprediction of elite boom performances
   - lightgbm_ceiling_p75, a 75th-percentile "upside" model for start/sit risk

3. Boom classification baselines and improved classifiers:
   - majority_class baseline
   - logistic_regression baseline
   - logistic_regression_balanced
   - lightgbm_default
   - lightgbm_balanced
   - lightgbm_balanced_tuned_threshold, which chooses a probability threshold
     on the 2023 validation season to improve F1 under class imbalance

The script is read-only with respect to Supabase. It writes CSV artifacts under
the local results directory.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class PositionConfig:
    position: str
    table: str
    high_score_threshold: float = 20.0


POSITION_CONFIGS = [
    PositionConfig("QB", "QBPointProjection", 20.0),
    PositionConfig("RB", "RBPointProjection", 20.0),
    PositionConfig("WRTE", "WRTEPointProjection", 20.0),
]


KEY_COLUMNS = ["player_id", "player_name", "season", "week", "team", "opponent_team"]
TARGET = "fantasy_points_ppr"
BOOM_THRESHOLDS = [10, 15, 20]


def build_engine():
    load_dotenv(PROJECT_ROOT / ".env")
    required = ["user", "password", "host", "port", "dbname"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing database settings in .env: {missing}")

    url = (
        f"postgresql+psycopg2://{os.getenv('user')}:{os.getenv('password')}"
        f"@{os.getenv('host')}:{os.getenv('port')}/{os.getenv('dbname')}?sslmode=require"
    )
    return create_engine(url, connect_args={"connect_timeout": 10})


def load_position_frame(engine, config: PositionConfig) -> pd.DataFrame:
    query = f'SELECT * FROM "{config.table}" WHERE season BETWEEN 2018 AND 2024'
    df = pd.read_sql(query, engine)
    return df[df["week"].between(1, 18)].copy()


def split_masks(df: pd.DataFrame):
    return {
        "train": df["season"].between(2018, 2022),
        "val": df["season"].eq(2023),
        "test": df["season"].eq(2024),
    }


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = KEY_COLUMNS + [TARGET]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).copy()
    return X.drop(columns=["fantasy_points_ppr_prev"], errors="ignore")


def build_feature_sets(X: pd.DataFrame) -> dict[str, list[str]]:
    player_rolling = [
        col
        for col in X.columns
        if (
            col.endswith("_lag1")
            or col.endswith("_roll3_mean")
            or col.endswith("_roll5_mean")
        )
        and "allowed" not in col
    ]
    previous_season = [
        col
        for col in X.columns
        if col.endswith("_prev") and col != "fantasy_points_ppr_prev"
    ]
    defense = [col for col in X.columns if "allowed" in col]

    return {
        "rolling_only": sorted(player_rolling),
        "rolling_plus_previous_season": sorted(set(player_rolling + previous_season)),
        "full_features": list(X.columns),
        "full_without_defense": [col for col in X.columns if col not in set(defense)],
    }


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    high_score_threshold: float,
) -> dict[str, float]:
    y_pred = np.asarray(y_pred)
    high_mask = y_true >= high_score_threshold
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "bias_pred_minus_actual": float(np.mean(y_pred - y_true)),
    }

    if high_mask.any():
        high_true = y_true.loc[high_mask]
        high_pred = y_pred[high_mask.to_numpy()]
        metrics.update(
            {
                "high_score_games": int(high_mask.sum()),
                "high_score_mae": mean_absolute_error(high_true, high_pred),
                "high_score_rmse": np.sqrt(mean_squared_error(high_true, high_pred)),
                "high_score_bias_pred_minus_actual": float(np.mean(high_pred - high_true)),
                "high_score_underprediction_rate": float(np.mean(high_pred < high_true)),
            }
        )
    else:
        metrics.update(
            {
                "high_score_games": 0,
                "high_score_mae": np.nan,
                "high_score_rmse": np.nan,
                "high_score_bias_pred_minus_actual": np.nan,
                "high_score_underprediction_rate": np.nan,
            }
        )
    return metrics


def fit_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weight: np.ndarray | None = None,
    objective: str = "regression",
    alpha: float | None = None,
):
    params = {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "objective": objective,
    }
    if alpha is not None:
        params["alpha"] = alpha

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        eval_metric="l2" if objective != "quantile" else "quantile",
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)],
    )
    return model


def high_score_weights(y_train: pd.Series, threshold: float) -> np.ndarray:
    percentile_90 = y_train.quantile(0.90)
    weights = np.ones(len(y_train), dtype=float)
    weights += 1.25 * (y_train >= threshold).to_numpy()
    weights += 0.75 * (y_train >= percentile_90).to_numpy()
    return weights


def run_regression_experiments(engine) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    results = []
    prediction_frames = []

    for config in POSITION_CONFIGS:
        print(f"Running regression experiments for {config.position}")
        df = load_position_frame(engine, config)
        masks = split_masks(df)
        X_all = base_features(df)
        y = df[TARGET]
        feature_sets = build_feature_sets(X_all)

        for feature_set_name, cols in feature_sets.items():
            if not cols:
                continue

            X = X_all[cols]
            X_train, y_train = X.loc[masks["train"]], y.loc[masks["train"]]
            X_val, y_val = X.loc[masks["val"]], y.loc[masks["val"]]
            X_test, y_test = X.loc[masks["test"]], y.loc[masks["test"]]

            linear = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LinearRegression()),
                ]
            )
            linear.fit(X_train, y_train)
            pred = linear.predict(X_test)
            results.append(
                {
                    "position": config.position,
                    "feature_set": feature_set_name,
                    "model": "linear_regression",
                    "num_features": len(cols),
                    **regression_metrics(y_test, pred, config.high_score_threshold),
                }
            )

            expected = fit_lgbm_regressor(X_train, y_train, X_val, y_val)
            pred = expected.predict(X_test, num_iteration=expected.best_iteration_)
            results.append(
                {
                    "position": config.position,
                    "feature_set": feature_set_name,
                    "model": "lightgbm_expected",
                    "num_features": len(cols),
                    "best_iteration": expected.best_iteration_,
                    **regression_metrics(y_test, pred, config.high_score_threshold),
                }
            )

            weighted = fit_lgbm_regressor(
                X_train,
                y_train,
                X_val,
                y_val,
                sample_weight=high_score_weights(y_train, config.high_score_threshold),
            )
            weighted_pred = weighted.predict(X_test, num_iteration=weighted.best_iteration_)
            results.append(
                {
                    "position": config.position,
                    "feature_set": feature_set_name,
                    "model": "lightgbm_boom_weighted",
                    "num_features": len(cols),
                    "best_iteration": weighted.best_iteration_,
                    **regression_metrics(y_test, weighted_pred, config.high_score_threshold),
                }
            )

            ceiling = fit_lgbm_regressor(
                X_train,
                y_train,
                X_val,
                y_val,
                objective="quantile",
                alpha=0.75,
            )
            ceiling_pred = ceiling.predict(X_test, num_iteration=ceiling.best_iteration_)
            results.append(
                {
                    "position": config.position,
                    "feature_set": feature_set_name,
                    "model": "lightgbm_ceiling_p75",
                    "num_features": len(cols),
                    "best_iteration": ceiling.best_iteration_,
                    **regression_metrics(y_test, ceiling_pred, config.high_score_threshold),
                }
            )

            if feature_set_name == "full_features":
                keys = df.loc[masks["test"], KEY_COLUMNS + [TARGET]].reset_index(drop=True)
                pred_frame = keys.copy()
                pred_frame["position_group"] = config.position
                pred_frame["lightgbm_expected"] = expected.predict(
                    X_test, num_iteration=expected.best_iteration_
                )
                pred_frame["lightgbm_boom_weighted"] = weighted_pred
                pred_frame["lightgbm_ceiling_p75"] = ceiling_pred
                pred_frame["expected_error"] = pred_frame["lightgbm_expected"] - pred_frame[TARGET]
                pred_frame["boom_weighted_error"] = (
                    pred_frame["lightgbm_boom_weighted"] - pred_frame[TARGET]
                )
                pred_frame["ceiling_error"] = pred_frame["lightgbm_ceiling_p75"] - pred_frame[TARGET]
                prediction_frames.append(pred_frame)

    results_df = pd.DataFrame(results)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    elite_examples = predictions_df[
        predictions_df["player_name"].isin(["J.Allen", "S.Barkley", "J.Chase"])
    ].sort_values(["position_group", "player_name", "week"])
    return results_df, predictions_df, elite_examples


def safe_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def safe_average_precision(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_score)


def classification_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": safe_auc(y_true, y_proba),
        "average_precision": safe_average_precision(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "positive_rate_actual": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(y_pred)),
    }


def best_f1_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, y_proba >= threshold, zero_division=0) for threshold in thresholds]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def fit_lgbm_classifier(X_train, y_train, X_val, y_val, balanced: bool = False):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        class_weight="balanced" if balanced else None,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return model


def run_classification_experiments(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = []
    prediction_frames = []

    for config in POSITION_CONFIGS:
        print(f"Running boom classification experiments for {config.position}")
        df = load_position_frame(engine, config)
        masks = split_masks(df)
        X = base_features(df)
        feature_cols = build_feature_sets(X)["full_features"]
        X = X[feature_cols]

        for threshold in BOOM_THRESHOLDS:
            y = (df[TARGET] >= threshold).astype(int)
            X_train, y_train = X.loc[masks["train"]], y.loc[masks["train"]]
            X_val, y_val = X.loc[masks["val"]], y.loc[masks["val"]]
            X_test, y_test = X.loc[masks["test"]], y.loc[masks["test"]]

            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X_train, y_train)
            pred = dummy.predict(X_test)
            proba = np.full(len(y_test), float(y_train.mean()))
            results.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "majority_class",
                    "decision_threshold": 0.5,
                    **classification_metrics(y_test, pred, proba),
                }
            )

            logistic = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, random_state=42)),
                ]
            )
            logistic.fit(X_train, y_train)
            proba = logistic.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            results.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "logistic_regression",
                    "decision_threshold": 0.5,
                    **classification_metrics(y_test, pred, proba),
                }
            )

            logistic_balanced = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            )
            logistic_balanced.fit(X_train, y_train)
            val_proba = logistic_balanced.predict_proba(X_val)[:, 1]
            tuned_threshold, val_f1 = best_f1_threshold(y_val, val_proba)
            proba = logistic_balanced.predict_proba(X_test)[:, 1]
            pred = (proba >= tuned_threshold).astype(int)
            results.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "logistic_regression_balanced_tuned_threshold",
                    "decision_threshold": tuned_threshold,
                    "validation_f1_at_threshold": val_f1,
                    **classification_metrics(y_test, pred, proba),
                }
            )

            lgb_default = fit_lgbm_classifier(X_train, y_train, X_val, y_val, balanced=False)
            proba = lgb_default.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            results.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "lightgbm_default",
                    "decision_threshold": 0.5,
                    "best_iteration": lgb_default.best_iteration_,
                    **classification_metrics(y_test, pred, proba),
                }
            )

            lgb_balanced = fit_lgbm_classifier(X_train, y_train, X_val, y_val, balanced=True)
            val_proba = lgb_balanced.predict_proba(X_val)[:, 1]
            tuned_threshold, val_f1 = best_f1_threshold(y_val, val_proba)
            proba = lgb_balanced.predict_proba(X_test)[:, 1]

            pred_default = (proba >= 0.5).astype(int)
            results.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "lightgbm_balanced_default_threshold",
                    "decision_threshold": 0.5,
                    "best_iteration": lgb_balanced.best_iteration_,
                    **classification_metrics(y_test, pred_default, proba),
                }
            )

            pred_tuned = (proba >= tuned_threshold).astype(int)
            results.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "lightgbm_balanced_tuned_threshold",
                    "decision_threshold": tuned_threshold,
                    "validation_f1_at_threshold": val_f1,
                    "best_iteration": lgb_balanced.best_iteration_,
                    **classification_metrics(y_test, pred_tuned, proba),
                }
            )

            keys = df.loc[masks["test"], KEY_COLUMNS + [TARGET]].reset_index(drop=True)
            pred_frame = keys.copy()
            pred_frame["position_group"] = config.position
            pred_frame["threshold"] = threshold
            pred_frame["actual_boom"] = y_test.reset_index(drop=True)
            pred_frame["lgbm_balanced_probability"] = proba
            pred_frame["lgbm_balanced_tuned_threshold"] = tuned_threshold
            pred_frame["lgbm_balanced_predicted_boom"] = pred_tuned
            prediction_frames.append(pred_frame)

    return pd.DataFrame(results), pd.concat(prediction_frames, ignore_index=True)


def round_float_columns(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()
    float_cols = rounded.select_dtypes(include=["float", "float64", "float32"]).columns
    rounded[float_cols] = rounded[float_cols].round(4)
    return rounded


def save_outputs(outputs: Iterable[tuple[pd.DataFrame, str]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for df, filename in outputs:
        path = RESULTS_DIR / filename
        round_float_columns(df).to_csv(path, index=False)
        print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "regression", "classification"],
        default="all",
        help="Which experiment suite to run.",
    )
    args = parser.parse_args()

    engine = build_engine()
    outputs: list[tuple[pd.DataFrame, str]] = []

    if args.mode in {"all", "regression"}:
        regression_results, regression_predictions, elite_examples = run_regression_experiments(
            engine
        )
        outputs.extend(
            [
                (regression_results, "regression_ablation_results.csv"),
                (regression_predictions, "regression_predictions_2024.csv"),
                (elite_examples, "elite_player_regression_examples_2024.csv"),
            ]
        )

    if args.mode in {"all", "classification"}:
        classification_results, classification_predictions = run_classification_experiments(engine)
        outputs.extend(
            [
                (classification_results, "boom_classification_baselines.csv"),
                (classification_predictions, "boom_predictions_2024.csv"),
            ]
        )

    save_outputs(outputs)


if __name__ == "__main__":
    main()
