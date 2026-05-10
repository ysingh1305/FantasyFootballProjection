"""
Improved controlled experiments for the fantasy football final paper.

This script keeps the same temporal design as model_alignment_experiments.py:
2018-2022 train, 2023 validation, and 2024 held-out test. The main difference is
that it adds leakage-safe features aimed at the weaknesses found in the first
experiment pass:

1. Player profile features:
   - prior career fantasy-point mean/std/ceiling
   - prior career boom rates at 10/15/20 PPR
   - season-to-date fantasy-point form
   - previous-season per-game production

2. Opportunity-share features:
   - player share of team/position rolling attempts, carries, targets, etc.
   - team/position rolling opportunity totals

3. Improved model variants:
   - validation-tuned LightGBM expected-points regressor
   - boom-weighted and quantile regressors
   - validation-selected expected/upside blends
   - balanced/tuned boom classifiers with optional isotonic calibration

The script reads from Supabase and writes CSV artifacts under results/. It never
prints database credentials.
"""

from __future__ import annotations

import argparse
import math
from itertools import product
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_alignment_experiments import (
    BOOM_THRESHOLDS,
    KEY_COLUMNS,
    POSITION_CONFIGS,
    RESULTS_DIR,
    TARGET,
    PositionConfig,
    base_features,
    best_f1_threshold,
    build_engine,
    classification_metrics,
    fit_lgbm_classifier,
    high_score_weights,
    regression_metrics,
    round_float_columns,
    split_masks,
)


EXPECTED_MODEL_COL = "enhanced_lightgbm_expected_tuned"
IBM_ESPN_PUBLISHED_METRICS = {
    "rmse": {
        "value": 6.78,
        "better_direction": "lower",
        "display_name": "RMSE",
    },
    "within_10_points": {
        "value": 0.882,
        "better_direction": "higher",
        "display_name": "Within 10 points",
    },
    "within_7_points": {
        "value": 0.710,
        "better_direction": "higher",
        "display_name": "Within 7 points",
    },
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def load_position_frame_with_history(engine, config: PositionConfig) -> pd.DataFrame:
    # 2017 is not used for training/testing, but it is valid history for 2018 rows.
    query = f'SELECT * FROM "{config.table}" WHERE season BETWEEN 2017 AND 2024'
    df = pd.read_sql(query, engine)
    df = df[df["week"].between(1, 18)].copy()
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week_index"] = df["week"].astype(float)
    df["early_season"] = (df["week"] <= 4).astype(int)
    df["late_season"] = (df["week"] >= 14).astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 18.0)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 18.0)
    return df


def add_previous_season_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "games_prev" not in df.columns:
        return df

    games_prev = pd.to_numeric(df["games_prev"], errors="coerce").fillna(0.0)
    df["had_prev_season"] = (games_prev > 0).astype(int)

    prev_cols = [
        col
        for col in df.columns
        if col.endswith("_prev") and col not in {"games_prev"}
    ]
    for col in prev_cols:
        base = col.removesuffix("_prev")
        df[f"{base}_prev_per_game"] = safe_divide(
            pd.to_numeric(df[col], errors="coerce"),
            games_prev,
        )

    return df


def add_player_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True).copy()
    y = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)

    player_groups = df.groupby("player_id", sort=False)
    prior_games = player_groups.cumcount()
    career_sum_before = y.groupby(df["player_id"]).cumsum() - y

    df["player_games_prior"] = prior_games
    df["player_fp_career_mean_prior"] = np.where(
        prior_games > 0,
        career_sum_before / prior_games.replace(0, np.nan),
        np.nan,
    )
    df["player_fp_career_std_prior"] = player_groups[TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=2).std()
    )
    df["player_fp_career_p75_prior"] = player_groups[TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=3).quantile(0.75)
    )
    df["player_fp_career_p90_prior"] = player_groups[TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=3).quantile(0.90)
    )
    df["player_fp_career_max_prior"] = player_groups[TARGET].transform(
        lambda s: s.shift(1).expanding(min_periods=1).max()
    )

    for window in (3, 5, 8):
        df[f"player_fp_roll{window}_mean"] = player_groups[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"player_fp_roll{window}_std"] = player_groups[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=2).std()
        )
        df[f"player_fp_roll{window}_max"] = player_groups[TARGET].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).max()
        )

    for threshold in BOOM_THRESHOLDS:
        indicator = (y >= threshold).astype(int)
        prior_booms = indicator.groupby(df["player_id"]).cumsum() - indicator
        df[f"player_boom{threshold}_rate_prior"] = np.where(
            prior_games > 0,
            prior_booms / prior_games.replace(0, np.nan),
            0.0,
        )

    season_groups = df.groupby(["player_id", "season"], sort=False)
    season_prior_games = season_groups.cumcount()
    season_sum_before = y.groupby([df["player_id"], df["season"]]).cumsum() - y
    df["player_games_this_season_prior"] = season_prior_games
    df["player_fp_season_mean_prior"] = np.where(
        season_prior_games > 0,
        season_sum_before / season_prior_games.replace(0, np.nan),
        np.nan,
    )

    for threshold in BOOM_THRESHOLDS:
        indicator = (y >= threshold).astype(int)
        prior_booms = (
            indicator.groupby([df["player_id"], df["season"]]).cumsum() - indicator
        )
        df[f"player_boom{threshold}_rate_this_season_prior"] = np.where(
            season_prior_games > 0,
            prior_booms / season_prior_games.replace(0, np.nan),
            0.0,
        )

    if "fantasy_points_ppr_prev_per_game" in df.columns:
        df["fp_roll3_vs_prev_per_game"] = safe_divide(
            df["player_fp_roll3_mean"],
            df["fantasy_points_ppr_prev_per_game"],
        )
        df["fp_career_mean_vs_prev_per_game"] = safe_divide(
            df["player_fp_career_mean_prior"],
            df["fantasy_points_ppr_prev_per_game"],
        )

    return df


def add_team_opportunity_share_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    opportunity_terms = (
        "attempts",
        "completions",
        "carries",
        "targets",
        "receptions",
        "passing_air_yards",
        "receiving_yards",
        "rushing_yards",
    )
    suffixes = ("_lag1", "_roll3_mean", "_roll5_mean")

    opportunity_cols = []
    for col in df.columns:
        if "allowed" in col or not col.endswith(suffixes):
            continue
        if col.startswith(opportunity_terms):
            opportunity_cols.append(col)

    group_cols = ["season", "week", "team"]
    for col in opportunity_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        team_total = values.groupby([df[c] for c in group_cols]).transform("sum")
        df[f"team_position_total_{col}"] = team_total
        df[f"team_position_share_{col}"] = safe_divide(values, team_total).fillna(0.0)

    return df


def add_usage_vs_previous_season_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in list(df.columns):
        if not col.endswith("_roll3_mean") or "allowed" in col:
            continue
        base = col.removesuffix("_roll3_mean")
        prev_pg_col = f"{base}_prev_per_game"
        if prev_pg_col in df.columns:
            df[f"{base}_roll3_vs_prev_per_game"] = safe_divide(
                pd.to_numeric(df[col], errors="coerce"),
                pd.to_numeric(df[prev_pg_col], errors="coerce"),
            )
    return df


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_previous_season_rate_features(df)
    df = add_player_profile_features(df)
    df = add_team_opportunity_share_features(df)
    df = add_usage_vs_previous_season_features(df)
    return df


def enhanced_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = base_features(df)
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns
    return X[numeric_cols].copy()


def fit_lgbm_regressor_with_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    sample_weight: np.ndarray | None = None,
    objective: str = "regression",
    alpha: float | None = None,
    params_override: dict | None = None,
):
    params = {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "objective": objective,
    }
    if alpha is not None:
        params["alpha"] = alpha
    if params_override:
        params.update(params_override)

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


def fit_validation_tuned_expected_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    candidates = []
    for num_leaves, min_child_samples, reg_lambda, colsample_bytree in product(
        [15, 31, 63],
        [20, 50],
        [0.0, 2.0],
        [0.7, 0.9],
    ):
        candidates.append(
            {
                "num_leaves": num_leaves,
                "min_child_samples": min_child_samples,
                "reg_lambda": reg_lambda,
                "colsample_bytree": colsample_bytree,
            }
        )

    best_model = None
    best_params = None
    best_rmse = math.inf

    for params in candidates:
        model = fit_lgbm_regressor_with_params(
            X_train,
            y_train,
            X_val,
            y_val,
            params_override=params,
        )
        pred = model.predict(X_val, num_iteration=model.best_iteration_)
        rmse = math.sqrt(mean_squared_error(y_val, pred))
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_params = params

    return best_model, best_params, best_rmse


def choose_blend_weight(
    y_val: pd.Series,
    expected_pred: np.ndarray,
    upside_pred: np.ndarray,
    *,
    objective: str,
    high_score_threshold: float,
) -> tuple[float, float]:
    best_weight = 0.0
    best_score = math.inf
    high_mask = y_val >= high_score_threshold

    for weight in np.arange(0.0, 1.01, 0.05):
        pred = (1.0 - weight) * expected_pred + weight * upside_pred
        if objective == "rmse":
            score = math.sqrt(mean_squared_error(y_val, pred))
        elif objective == "high_score_mae" and high_mask.any():
            score = mean_absolute_error(y_val.loc[high_mask], pred[high_mask.to_numpy()])
        else:
            score = mean_absolute_error(y_val, pred)

        if score < best_score:
            best_score = score
            best_weight = float(weight)

    return best_weight, float(best_score)


def regression_metric_summary(
    predictions: pd.DataFrame,
    prediction_col: str = EXPECTED_MODEL_COL,
) -> dict[str, float]:
    clean = predictions[[TARGET, prediction_col]].dropna()
    y_true = clean[TARGET].to_numpy(float)
    y_pred = clean[prediction_col].to_numpy(float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "within_10_points": float(np.mean(abs_error <= 10.0)),
        "within_7_points": float(np.mean(abs_error <= 7.0)),
    }


def bootstrap_regression_metric_intervals(
    predictions: pd.DataFrame,
    prediction_col: str = EXPECTED_MODEL_COL,
    *,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 439,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], dict[str, int | str | float]]:
    """Estimate test-metric uncertainty by resampling players with replacement."""
    required_cols = ["player_id", TARGET, prediction_col]
    clean = predictions[required_cols].dropna().reset_index(drop=True)
    if clean.empty:
        raise ValueError("Cannot bootstrap confidence intervals from an empty prediction frame.")

    player_ids = clean["player_id"].to_numpy()
    players = np.unique(player_ids)
    player_index = {player: np.flatnonzero(player_ids == player) for player in players}
    metric_names = ["rmse", "within_10_points", "within_7_points"]
    lower_pct = 100.0 * (1.0 - confidence_level) / 2.0
    upper_pct = 100.0 - lower_pct

    rng = np.random.default_rng(seed)
    boot = np.empty((n_bootstrap, len(metric_names)))
    for sample_idx in range(n_bootstrap):
        sampled_players = rng.choice(players, size=len(players), replace=True)
        sampled_rows = np.concatenate([player_index[player] for player in sampled_players])
        sample_metrics = regression_metric_summary(clean.iloc[sampled_rows], prediction_col)
        boot[sample_idx] = [sample_metrics[name] for name in metric_names]

    point_estimates = regression_metric_summary(clean, prediction_col)
    percentiles = np.percentile(boot, [lower_pct, upper_pct], axis=0)
    intervals = {
        metric: (float(percentiles[0, idx]), float(percentiles[1, idx]))
        for idx, metric in enumerate(metric_names)
    }
    metadata = {
        "confidence_level": confidence_level,
        "bootstrap_unit": "player_id",
        "n_bootstrap": n_bootstrap,
        "n_rows": len(clean),
        "n_players": len(players),
    }
    return point_estimates, intervals, metadata


def build_regression_confidence_outputs(
    predictions: pd.DataFrame,
    *,
    prediction_col: str = EXPECTED_MODEL_COL,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 439,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interval_rows = []
    top500_player_ids = (
        predictions.groupby("player_id")[TARGET]
        .sum()
        .sort_values(ascending=False)
        .head(500)
        .index
    )
    top500_predictions = predictions[predictions["player_id"].isin(top500_player_ids)]
    scopes = [
        ("overall", predictions),
        ("top500_total_ppr", top500_predictions),
    ]
    scopes.extend(
        (position, group)
        for position, group in predictions.groupby("position_group", sort=True)
    )

    estimates_by_scope = {}
    intervals_by_scope = {}
    metadata_by_scope = {}
    for scope_idx, (scope, frame) in enumerate(scopes):
        estimates, intervals, metadata = bootstrap_regression_metric_intervals(
            frame,
            prediction_col,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            seed=seed + scope_idx,
        )
        estimates_by_scope[scope] = estimates
        intervals_by_scope[scope] = intervals
        metadata_by_scope[scope] = metadata

        for metric, estimate in estimates.items():
            low, high = intervals[metric]
            interval_rows.append(
                {
                    "model": prediction_col,
                    "scope": scope,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_lower": low,
                    "ci_upper": high,
                    **metadata,
                }
            )

    comparison_rows = []
    for scope in ("overall", "top500_total_ppr"):
        for metric, published in IBM_ESPN_PUBLISHED_METRICS.items():
            estimate = estimates_by_scope[scope][metric]
            low, high = intervals_by_scope[scope][metric]
            published_value = published["value"]
            better_direction = published["better_direction"]
            if better_direction == "lower":
                relation = (
                    "our_ci_entirely_better"
                    if high < published_value
                    else "published_point_inside_or_better_than_our_ci"
                )
            else:
                relation = (
                    "our_ci_entirely_better"
                    if low > published_value
                    else "published_point_inside_or_better_than_our_ci"
                )

            comparison_rows.append(
                {
                    "comparison": "IBM/ESPN published point estimate vs our enhanced 2024 test metric",
                    "scope": scope,
                    "metric": metric,
                    "display_name": published["display_name"],
                    "better_direction": better_direction,
                    "ibm_espn_published_estimate": published_value,
                    "our_estimate": estimate,
                    "our_ci_lower": low,
                    "our_ci_upper": high,
                    "delta_ours_minus_ibm_espn": estimate - published_value,
                    "ci_relation_to_published_point_estimate": relation,
                    **metadata_by_scope[scope],
                }
            )

    return pd.DataFrame(interval_rows), pd.DataFrame(comparison_rows)


def build_original_vs_enhanced_bootstrap(
    original_predictions: pd.DataFrame,
    enhanced_predictions: pd.DataFrame,
    *,
    original_col: str = "lightgbm_expected",
    enhanced_col: str = EXPECTED_MODEL_COL,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 440,
) -> pd.DataFrame:
    join_cols = KEY_COLUMNS + [TARGET, "position_group"]
    original = original_predictions[join_cols + [original_col]]
    enhanced = enhanced_predictions[join_cols + [enhanced_col]]
    merged = original.merge(enhanced, on=join_cols, how="inner")
    if merged.empty:
        raise ValueError("Cannot bootstrap original-vs-enhanced comparison from an empty merge.")

    player_ids = merged["player_id"].to_numpy()
    players = np.unique(player_ids)
    player_index = {player: np.flatnonzero(player_ids == player) for player in players}
    metric_names = ["rmse", "mae", "within_10_points", "within_7_points"]
    lower_pct = 100.0 * (1.0 - confidence_level) / 2.0
    upper_pct = 100.0 - lower_pct

    def paired_metrics(index: np.ndarray) -> dict[str, tuple[float, float, float]]:
        frame = merged.iloc[index]
        y_true = frame[TARGET].to_numpy(float)
        original_error = frame[original_col].to_numpy(float) - y_true
        enhanced_error = frame[enhanced_col].to_numpy(float) - y_true
        original_abs = np.abs(original_error)
        enhanced_abs = np.abs(enhanced_error)
        original_values = {
            "rmse": float(np.sqrt(np.mean(original_error * original_error))),
            "mae": float(np.mean(original_abs)),
            "within_10_points": float(np.mean(original_abs <= 10.0)),
            "within_7_points": float(np.mean(original_abs <= 7.0)),
        }
        enhanced_values = {
            "rmse": float(np.sqrt(np.mean(enhanced_error * enhanced_error))),
            "mae": float(np.mean(enhanced_abs)),
            "within_10_points": float(np.mean(enhanced_abs <= 10.0)),
            "within_7_points": float(np.mean(enhanced_abs <= 7.0)),
        }
        improvement = {
            "rmse": original_values["rmse"] - enhanced_values["rmse"],
            "mae": original_values["mae"] - enhanced_values["mae"],
            "within_10_points": enhanced_values["within_10_points"] - original_values["within_10_points"],
            "within_7_points": enhanced_values["within_7_points"] - original_values["within_7_points"],
        }
        return {
            metric: (original_values[metric], enhanced_values[metric], improvement[metric])
            for metric in metric_names
        }

    rng = np.random.default_rng(seed)
    boot = np.empty((n_bootstrap, len(metric_names)))
    for sample_idx in range(n_bootstrap):
        sampled_players = rng.choice(players, size=len(players), replace=True)
        sampled_rows = np.concatenate([player_index[player] for player in sampled_players])
        sample_metrics = paired_metrics(sampled_rows)
        boot[sample_idx] = [sample_metrics[name][2] for name in metric_names]

    point_estimates = paired_metrics(np.arange(len(merged)))
    percentiles = np.percentile(boot, [lower_pct, upper_pct], axis=0)
    rows = []
    for metric_idx, metric in enumerate(metric_names):
        original_estimate, enhanced_estimate, improvement_estimate = point_estimates[metric]
        rows.append(
            {
                "metric": metric,
                "original_estimate": original_estimate,
                "enhanced_estimate": enhanced_estimate,
                "improvement_estimate": improvement_estimate,
                "improvement_ci_lower": float(percentiles[0, metric_idx]),
                "improvement_ci_upper": float(percentiles[1, metric_idx]),
                "confidence_level": confidence_level,
                "bootstrap_unit": "player_id",
                "n_bootstrap": n_bootstrap,
                "n_rows": len(merged),
                "n_players": len(players),
            }
        )
    return pd.DataFrame(rows)


def run_improved_regression(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_frames = []

    for config in POSITION_CONFIGS:
        print(f"Running improved regression for {config.position}")
        df = add_enhanced_features(load_position_frame_with_history(engine, config))
        masks = split_masks(df)
        X = enhanced_feature_matrix(df)
        y = df[TARGET]

        X_train, y_train = X.loc[masks["train"]], y.loc[masks["train"]]
        X_val, y_val = X.loc[masks["val"]], y.loc[masks["val"]]
        X_test, y_test = X.loc[masks["test"]], y.loc[masks["test"]]

        expected, best_params, val_rmse = fit_validation_tuned_expected_regressor(
            X_train,
            y_train,
            X_val,
            y_val,
        )
        val_expected_pred = expected.predict(X_val, num_iteration=expected.best_iteration_)
        test_expected_pred = expected.predict(X_test, num_iteration=expected.best_iteration_)
        rows.append(
            {
                "position": config.position,
                "model": "enhanced_lightgbm_expected_tuned",
                "num_features": X.shape[1],
                "best_iteration": expected.best_iteration_,
                "validation_rmse": val_rmse,
                "best_params": best_params,
                **regression_metrics(y_test, test_expected_pred, config.high_score_threshold),
            }
        )

        weighted = fit_lgbm_regressor_with_params(
            X_train,
            y_train,
            X_val,
            y_val,
            sample_weight=high_score_weights(y_train, config.high_score_threshold),
        )
        test_weighted_pred = weighted.predict(X_test, num_iteration=weighted.best_iteration_)
        rows.append(
            {
                "position": config.position,
                "model": "enhanced_lightgbm_boom_weighted",
                "num_features": X.shape[1],
                "best_iteration": weighted.best_iteration_,
                **regression_metrics(y_test, test_weighted_pred, config.high_score_threshold),
            }
        )

        quantile_predictions = {}
        val_quantile_predictions = {}
        for alpha in (0.50, 0.75, 0.90):
            quantile = fit_lgbm_regressor_with_params(
                X_train,
                y_train,
                X_val,
                y_val,
                objective="quantile",
                alpha=alpha,
            )
            val_pred = quantile.predict(X_val, num_iteration=quantile.best_iteration_)
            test_pred = quantile.predict(X_test, num_iteration=quantile.best_iteration_)
            suffix = f"p{int(alpha * 100)}"
            val_quantile_predictions[suffix] = val_pred
            quantile_predictions[suffix] = test_pred
            rows.append(
                {
                    "position": config.position,
                    "model": f"enhanced_lightgbm_quantile_{suffix}",
                    "num_features": X.shape[1],
                    "best_iteration": quantile.best_iteration_,
                    **regression_metrics(y_test, test_pred, config.high_score_threshold),
                }
            )

        rmse_weight, rmse_score = choose_blend_weight(
            y_val,
            val_expected_pred,
            val_quantile_predictions["p75"],
            objective="rmse",
            high_score_threshold=config.high_score_threshold,
        )
        rmse_blend_pred = (1.0 - rmse_weight) * test_expected_pred + rmse_weight * quantile_predictions["p75"]
        rows.append(
            {
                "position": config.position,
                "model": "enhanced_blend_expected_p75_val_rmse",
                "num_features": X.shape[1],
                "blend_weight": rmse_weight,
                "validation_blend_score": rmse_score,
                **regression_metrics(y_test, rmse_blend_pred, config.high_score_threshold),
            }
        )

        tail_weight, tail_score = choose_blend_weight(
            y_val,
            val_expected_pred,
            val_quantile_predictions["p90"],
            objective="high_score_mae",
            high_score_threshold=config.high_score_threshold,
        )
        tail_blend_pred = (1.0 - tail_weight) * test_expected_pred + tail_weight * quantile_predictions["p90"]
        rows.append(
            {
                "position": config.position,
                "model": "enhanced_blend_expected_p90_val_tail_mae",
                "num_features": X.shape[1],
                "blend_weight": tail_weight,
                "validation_blend_score": tail_score,
                **regression_metrics(y_test, tail_blend_pred, config.high_score_threshold),
            }
        )

        keys = df.loc[masks["test"], KEY_COLUMNS + [TARGET]].reset_index(drop=True)
        pred_frame = keys.copy()
        pred_frame["position_group"] = config.position
        pred_frame["enhanced_lightgbm_expected_tuned"] = test_expected_pred
        pred_frame["enhanced_lightgbm_boom_weighted"] = test_weighted_pred
        pred_frame["enhanced_lightgbm_quantile_p50"] = quantile_predictions["p50"]
        pred_frame["enhanced_lightgbm_quantile_p75"] = quantile_predictions["p75"]
        pred_frame["enhanced_lightgbm_quantile_p90"] = quantile_predictions["p90"]
        pred_frame["enhanced_blend_expected_p75_val_rmse"] = rmse_blend_pred
        pred_frame["enhanced_blend_expected_p90_val_tail_mae"] = tail_blend_pred
        for col in [
            "enhanced_lightgbm_expected_tuned",
            "enhanced_lightgbm_boom_weighted",
            "enhanced_lightgbm_quantile_p50",
            "enhanced_lightgbm_quantile_p75",
            "enhanced_lightgbm_quantile_p90",
            "enhanced_blend_expected_p75_val_rmse",
            "enhanced_blend_expected_p90_val_tail_mae",
        ]:
            pred_frame[f"{col}_error"] = pred_frame[col] - pred_frame[TARGET]
        prediction_frames.append(pred_frame)

    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


def fit_enhanced_logistic_classifier(X_train, y_train):
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    C=0.5,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def run_improved_classification(engine) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_frames = []

    for config in POSITION_CONFIGS:
        print(f"Running improved classification for {config.position}")
        df = add_enhanced_features(load_position_frame_with_history(engine, config))
        masks = split_masks(df)
        X = enhanced_feature_matrix(df)

        for threshold in BOOM_THRESHOLDS:
            y = (df[TARGET] >= threshold).astype(int)
            X_train, y_train = X.loc[masks["train"]], y.loc[masks["train"]]
            X_val, y_val = X.loc[masks["val"]], y.loc[masks["val"]]
            X_test, y_test = X.loc[masks["test"]], y.loc[masks["test"]]

            logistic = fit_enhanced_logistic_classifier(X_train, y_train)
            logistic_val_proba = logistic.predict_proba(X_val)[:, 1]
            tuned_threshold, val_f1 = best_f1_threshold(y_val, logistic_val_proba)
            logistic_proba = logistic.predict_proba(X_test)[:, 1]
            pred = (logistic_proba >= tuned_threshold).astype(int)
            rows.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "enhanced_logistic_balanced_tuned_threshold",
                    "num_features": X.shape[1],
                    "decision_threshold": tuned_threshold,
                    "validation_f1_at_threshold": val_f1,
                    **classification_metrics(y_test, pred, logistic_proba),
                }
            )

            lgb_balanced = fit_lgbm_classifier(X_train, y_train, X_val, y_val, balanced=True)
            val_proba = lgb_balanced.predict_proba(X_val)[:, 1]
            tuned_threshold, val_f1 = best_f1_threshold(y_val, val_proba)
            proba = lgb_balanced.predict_proba(X_test)[:, 1]
            pred = (proba >= tuned_threshold).astype(int)
            rows.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "enhanced_lightgbm_balanced_tuned_threshold",
                    "num_features": X.shape[1],
                    "decision_threshold": tuned_threshold,
                    "validation_f1_at_threshold": val_f1,
                    "best_iteration": lgb_balanced.best_iteration_,
                    **classification_metrics(y_test, pred, proba),
                }
            )

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(val_proba, y_val)
            val_calibrated = calibrator.predict(val_proba)
            calibrated_threshold, calibrated_val_f1 = best_f1_threshold(y_val, val_calibrated)
            calibrated_proba = calibrator.predict(proba)
            calibrated_pred = (calibrated_proba >= calibrated_threshold).astype(int)
            rows.append(
                {
                    "position": config.position,
                    "threshold": threshold,
                    "model": "enhanced_lightgbm_balanced_isotonic_tuned_threshold",
                    "num_features": X.shape[1],
                    "decision_threshold": calibrated_threshold,
                    "validation_f1_at_threshold": calibrated_val_f1,
                    "best_iteration": lgb_balanced.best_iteration_,
                    **classification_metrics(y_test, calibrated_pred, calibrated_proba),
                }
            )

            keys = df.loc[masks["test"], KEY_COLUMNS + [TARGET]].reset_index(drop=True)
            pred_frame = keys.copy()
            pred_frame["position_group"] = config.position
            pred_frame["threshold"] = threshold
            pred_frame["actual_boom"] = y_test.reset_index(drop=True)
            pred_frame["enhanced_logistic_probability"] = logistic_proba
            pred_frame["enhanced_lgbm_probability"] = proba
            pred_frame["enhanced_lgbm_isotonic_probability"] = calibrated_proba
            pred_frame["enhanced_lgbm_isotonic_threshold"] = calibrated_threshold
            pred_frame["enhanced_lgbm_isotonic_predicted_boom"] = calibrated_pred
            prediction_frames.append(pred_frame)

    return pd.DataFrame(rows), pd.concat(prediction_frames, ignore_index=True)


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
        choices=["all", "regression", "classification", "confidence"],
        default="all",
        help=(
            "Which experiment suite to run. Use 'confidence' to recompute "
            "bootstrap intervals from existing improved regression predictions."
        ),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10000,
        help="Number of player-cluster bootstrap samples for regression confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=439,
        help="Random seed for player-cluster bootstrap confidence intervals.",
    )
    args = parser.parse_args()

    outputs: list[tuple[pd.DataFrame, str]] = []
    engine = None
    if args.mode in {"all", "regression", "classification"}:
        engine = build_engine()

    if args.mode in {"all", "regression"}:
        if engine is None:
            raise RuntimeError("Database engine was not initialized.")
        regression_results, regression_predictions = run_improved_regression(engine)
        confidence_intervals, espn_comparison = build_regression_confidence_outputs(
            regression_predictions,
            n_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        outputs.extend(
            [
                (regression_results, "improved_regression_results.csv"),
                (regression_predictions, "improved_regression_predictions_2024.csv"),
                (confidence_intervals, "improved_regression_metric_confidence_intervals.csv"),
                (espn_comparison, "improved_regression_espn_comparison.csv"),
            ]
        )
        original_predictions_path = RESULTS_DIR / "regression_predictions_2024.csv"
        if original_predictions_path.exists():
            original_predictions = pd.read_csv(original_predictions_path)
            paired_bootstrap = build_original_vs_enhanced_bootstrap(
                original_predictions,
                regression_predictions,
                n_bootstrap=args.bootstrap_samples,
                seed=args.bootstrap_seed + 1,
            )
            outputs.append(
                (paired_bootstrap, "improved_vs_original_regression_bootstrap.csv")
            )

    if args.mode in {"all", "classification"}:
        if engine is None:
            raise RuntimeError("Database engine was not initialized.")
        classification_results, classification_predictions = run_improved_classification(engine)
        outputs.extend(
            [
                (classification_results, "improved_boom_classification_results.csv"),
                (classification_predictions, "improved_boom_predictions_2024.csv"),
            ]
        )

    if args.mode == "confidence":
        predictions_path = RESULTS_DIR / "improved_regression_predictions_2024.csv"
        regression_predictions = pd.read_csv(predictions_path)
        confidence_intervals, espn_comparison = build_regression_confidence_outputs(
            regression_predictions,
            n_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        )
        original_predictions = pd.read_csv(RESULTS_DIR / "regression_predictions_2024.csv")
        paired_bootstrap = build_original_vs_enhanced_bootstrap(
            original_predictions,
            regression_predictions,
            n_bootstrap=args.bootstrap_samples,
            seed=args.bootstrap_seed + 1,
        )
        outputs.extend(
            [
                (confidence_intervals, "improved_regression_metric_confidence_intervals.csv"),
                (espn_comparison, "improved_regression_espn_comparison.csv"),
                (paired_bootstrap, "improved_vs_original_regression_bootstrap.csv"),
            ]
        )

    save_outputs(outputs)


if __name__ == "__main__":
    main()
