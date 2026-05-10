from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "figures"
GEN_DIR = PAPER / "generated"
sys.path.insert(0, str(EXPERIMENTS))

from improved_model_experiments import (  # noqa: E402
    TARGET,
    add_enhanced_features,
    enhanced_feature_matrix,
    fit_lgbm_regressor_with_params,
    load_position_frame_with_history,
)
from model_alignment_experiments import POSITION_CONFIGS, build_engine, split_masks  # noqa: E402


def feature_group(feature: str) -> str:
    if "allowed" in feature:
        return "Defensive matchup"
    if "team_position_share" in feature or "team_position_total" in feature:
        return "Opportunity share"
    if "career" in feature or "boom" in feature or "player_fp_" in feature:
        return "Player profile"
    if feature.endswith("_prev") or "prev_per_game" in feature or "had_prev_season" in feature:
        return "Prior season"
    if feature.endswith("_lag1") or "_roll3_" in feature or "_roll5_" in feature or "_roll8_" in feature:
        return "Recent form"
    if feature.startswith("week") or feature in {"early_season", "late_season"}:
        return "Temporal"
    return "Other"


def clean_feature_name(name: str) -> str:
    replacements = {
        "_": "\\_",
        "fantasy\\_points\\_ppr": "fp",
        "player\\_fp": "player fp",
        "team\\_position\\_share": "team share",
        "team\\_position\\_total": "team total",
        "career": "career",
        "prior": "prior",
        "roll3\\_mean": "roll3 mean",
        "roll5\\_mean": "roll5 mean",
        "lag1": "lag1",
    }
    out = name
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def booktabs(headers, rows, align):
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(str(x) for x in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def write_table(path: Path, tabular: str, caption: str, label: str):
    path.write_text(
        "\\begin{table}[!htbp]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n"
        "\\footnotesize\n"
        + tabular
        + "\n\\end{table}\n",
        encoding="utf-8",
    )


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(ROOT / "results/improved_regression_results.csv")
    best = results[results["model"].eq("enhanced_lightgbm_expected_tuned")].set_index("position")
    engine = build_engine()

    feature_rows = []
    group_rows = []

    for config in POSITION_CONFIGS:
        print(f"Training feature-importance model for {config.position}")
        df = add_enhanced_features(load_position_frame_with_history(engine, config))
        masks = split_masks(df)
        X = enhanced_feature_matrix(df)
        y = df[TARGET]
        params = ast.literal_eval(best.loc[config.position, "best_params"])

        model = fit_lgbm_regressor_with_params(
            X.loc[masks["train"]],
            y.loc[masks["train"]],
            X.loc[masks["val"]],
            y.loc[masks["val"]],
            params_override=params,
        )
        gains = model.booster_.feature_importance(importance_type="gain")
        imp = pd.DataFrame(
            {
                "position": config.position,
                "feature": X.columns,
                "gain": gains,
            }
        )
        total = imp["gain"].sum()
        imp["normalized_gain"] = np.where(total > 0, imp["gain"] / total, 0.0)
        imp["group"] = imp["feature"].map(feature_group)
        feature_rows.append(imp)
        grouped = imp.groupby("group", as_index=False)["normalized_gain"].sum()
        grouped["position"] = config.position
        group_rows.append(grouped)

    all_features = pd.concat(feature_rows, ignore_index=True)
    all_groups = pd.concat(group_rows, ignore_index=True)
    all_features.to_csv(GEN_DIR / "feature_importance.csv", index=False)
    all_groups.to_csv(GEN_DIR / "feature_group_importance.csv", index=False)

    top_rows = []
    for pos in ["QB", "RB", "WRTE"]:
        top = (
            all_features[all_features["position"].eq(pos)]
            .sort_values("normalized_gain", ascending=False)
            .head(5)
        )
        top_rows.append(
            [
                pos.replace("WRTE", "WR/TE"),
                "; ".join(
                    f"{clean_feature_name(row.feature)} ({100 * row.normalized_gain:.1f}\\%)"
                    for row in top.itertuples()
                ),
            ]
        )
    write_table(
        GEN_DIR / "feature_importance_table.tex",
        booktabs(["Pos.", "Top gain-based features"], top_rows, "lp{0.76\\linewidth}"),
        "Top enhanced expected-model LightGBM features by gain on the training/validation fit. Percentages are normalized within each position model.",
        "tab:importance",
    )

    order = [
        "Recent form",
        "Player profile",
        "Opportunity share",
        "Prior season",
        "Defensive matchup",
        "Temporal",
        "Other",
    ]
    pivot = (
        all_groups.pivot_table(
            index="group",
            columns="position",
            values="normalized_gain",
            fill_value=0.0,
        )
        .reindex(order)
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    x = np.arange(len(pivot.index))
    width = 0.24
    colors = {"QB": "#386cb0", "RB": "#1b9e77", "WRTE": "#d95f02"}
    for i, pos in enumerate(["QB", "RB", "WRTE"]):
        ax.bar(x + (i - 1) * width, 100 * pivot[pos], width, color=colors[pos], label=pos.replace("WRTE", "WR/TE"))
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=18, ha="right")
    ax.set_ylabel("Normalized gain (%)")
    ax.set_title("Feature importance by group")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_group_importance.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "feature_group_importance.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
