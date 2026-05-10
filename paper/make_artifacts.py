from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIG_DIR = PAPER / "figures"
GEN_DIR = PAPER / "generated"


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def tex_table(path, tabular, caption, label):
    path.write_text(
        "\\begin{table}[!htbp]\n"
        "\\caption{" + caption + "}\n"
        "\\label{" + label + "}\n"
        "\\centering\n"
        "\\footnotesize\n"
        + tabular
        + "\n\\end{table}\n",
        encoding="utf-8",
    )


def booktabs(headers, rows, align=None):
    if align is None:
        align = "l" + "r" * (len(headers) - 1)
    out = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    out.append(" & ".join(headers) + " \\\\")
    out.append("\\midrule")
    for row in rows:
        out.append(" & ".join(str(x) for x in row) + " \\\\")
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out)


def fmt(x, digits=2):
    return f"{x:.{digits}f}"


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%"


def metric_cell(value, low, high, *, percent=False):
    if percent:
        return f"\\good{{{100 * value:.1f}\\%}} [{100 * low:.1f}, {100 * high:.1f}]\\%"
    return f"\\good{{{value:.2f}}} [{low:.2f}, {high:.2f}]"


def dataset_summary():
    # Avoid any dependency on the discarded prior project folder. The full table
    # row counts are the verified Supabase counts supplied with the project
    # brief; the held-out 2024 summaries come from active result artifacts.
    table_rows = {
        "QB": 4857,
        "RB": 11668,
        "WR/TE": 28205,
    }
    pred = pd.read_csv(ROOT / "results/improved_regression_predictions_2024.csv")
    position_map = {
        "QB": "QB",
        "RB": "RB",
        "WRTE": "WR/TE",
    }
    rows = []
    for pos_key, pos_label in position_map.items():
        df = pred[pred["position_group"].eq(pos_key)].copy()
        rows.append(
            [
                pos_label,
                f"{table_rows[pos_label]:,}",
                f"{len(df):,}",
                f"{df['player_id'].nunique():,}",
                fmt(df["fantasy_points_ppr"].mean()),
                pct((df["fantasy_points_ppr"] >= 20).mean()),
            ]
        )
    tab = booktabs(
        [
            "Pos.",
            "Table rows",
            "2024 rows",
            "2024 players",
            "2024 mean",
            "2024 20+",
        ],
        rows,
        align="lrrrrr",
    )
    tex_table(
        GEN_DIR / "dataset_summary.tex",
        tab,
        "Dataset summary by position group. Table row counts are the verified Supabase counts for 2017--2024; the held-out summaries are computed from active 2024 result artifacts. The 2017 season is retained only as pre-2018 history for leakage-safe features; model training uses 2018--2022, validation uses 2023, and the reported test set is 2024.",
        "tab:dataset",
    )


def regression_tables():
    orig = pd.read_csv(ROOT / "results/regression_predictions_2024.csv")
    enh = pd.read_csv(ROOT / "results/improved_regression_predictions_2024.csv")

    def metrics(df, pred_col):
        y = df["fantasy_points_ppr"].to_numpy()
        p = df[pred_col].to_numpy()
        high = y >= 20
        return {
            "mae": mae(y, p),
            "rmse": rmse(y, p),
            "r2": r2(y, p),
            "within7": float(np.mean(np.abs(y - p) <= 7)),
            "within10": float(np.mean(np.abs(y - p) <= 10)),
            "high_mae": mae(y[high], p[high]),
            "high_under": float(np.mean(p[high] < y[high])),
        }

    model_rows = []
    for name, df, col in [
        ("Original LightGBM expected", orig, "lightgbm_expected"),
        ("Enhanced LightGBM expected", enh, "enhanced_lightgbm_expected_tuned"),
    ]:
        m = metrics(df, col)
        model_rows.append(
            [
                name,
                fmt(m["mae"]),
                fmt(m["rmse"]),
                fmt(m["r2"], 3),
                pct(m["within7"]),
                pct(m["within10"]),
                fmt(m["high_mae"]),
            ]
        )
    tex_table(
        GEN_DIR / "regression_overall.tex",
        booktabs(
            ["Model", "MAE", "RMSE", "$R^2$", "$|e|\\leq7$", "$|e|\\leq10$", "20+ MAE"],
            model_rows,
            align="lrrrrrr",
        ),
        "Overall held-out 2024 point-projection performance across all positions.",
        "tab:regression-overall",
    )

    pos_rows = []
    for pos in ["QB", "RB", "WRTE"]:
        o = orig[orig["position_group"] == pos]
        e = enh[enh["position_group"] == pos]
        o_rmse = rmse(o["fantasy_points_ppr"], o["lightgbm_expected"])
        e_rmse = rmse(e["fantasy_points_ppr"], e["enhanced_lightgbm_expected_tuned"])
        e_mae = mae(e["fantasy_points_ppr"], e["enhanced_lightgbm_expected_tuned"])
        high = e["fantasy_points_ppr"] >= 20
        pos_rows.append(
            [
                pos.replace("WRTE", "WR/TE"),
                f"{len(e):,}",
                fmt(o_rmse),
                fmt(e_rmse),
                fmt(o_rmse - e_rmse),
                fmt(e_mae),
                fmt(mae(e.loc[high, "fantasy_points_ppr"], e.loc[high, "enhanced_lightgbm_expected_tuned"])),
            ]
        )
    tex_table(
        GEN_DIR / "regression_by_position.tex",
        booktabs(
            ["Pos.", "Rows", "Orig. RMSE", "Enh. RMSE", "$\\Delta$", "Enh. MAE", "Enh. 20+ MAE"],
            pos_rows,
            align="lrrrrrr",
        ),
        "Expected-point regression results by position group on the 2024 test season.",
        "tab:regression-position",
    )

    tail_rows = []
    for name, col in [
        ("Expected", "enhanced_lightgbm_expected_tuned"),
        ("Boom-weighted", "enhanced_lightgbm_boom_weighted"),
        ("Quantile p75", "enhanced_lightgbm_quantile_p75"),
        ("Quantile p90", "enhanced_lightgbm_quantile_p90"),
        ("Expected/p90 blend", "enhanced_blend_expected_p90_val_tail_mae"),
    ]:
        m = metrics(enh, col)
        tail_rows.append(
            [
                name,
                fmt(m["mae"]),
                fmt(m["rmse"]),
                fmt(m["high_mae"]),
                pct(m["high_under"]),
                pct(m["within10"]),
            ]
        )
    tex_table(
        GEN_DIR / "tail_quantile.tex",
        booktabs(
            ["Model", "MAE", "RMSE", "20+ MAE", "20+ under", "$|e|\\leq10$"],
            tail_rows,
            align="lrrrrr",
        ),
        "Tail-risk comparison on 2024. Quantile models are useful as ceiling estimates, but they are not replacements for calibrated expected-point projections.",
        "tab:tail",
    )

    comparison_path = ROOT / "results/improved_regression_espn_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(
            "Missing improved_regression_espn_comparison.csv. "
            "Run `python experiments/improved_model_experiments.py --mode confidence` first."
        )

    comparison = pd.read_csv(comparison_path).set_index(["scope", "metric"])
    espn_rows = []
    for metric in ["rmse", "within_10_points", "within_7_points"]:
        row = comparison.loc[("overall", metric)]
        top500 = comparison.loc[("top500_total_ppr", metric)]
        espn_rows.append(
            (
                row["display_name"].replace("points", "pts"),
                row["better_direction"].title(),
                row["ibm_espn_published_estimate"],
                row["our_estimate"],
                row["our_ci_lower"],
                row["our_ci_upper"],
                top500["our_estimate"],
                top500["our_ci_lower"],
                top500["our_ci_upper"],
                "\\%" if metric.startswith("within") else "",
            )
        )
    rows = []
    for metric, better, espn_value, our_value, low, high, top_value, top_low, top_high, suffix in espn_rows:
        if suffix:
            espn_cell = f"\\bad{{{100 * espn_value:.1f}{suffix}}}"
            ours_cell = metric_cell(our_value, low, high, percent=True)
            top_cell = metric_cell(top_value, top_low, top_high, percent=True)
        else:
            espn_cell = f"\\bad{{{espn_value:.2f}}}"
            ours_cell = metric_cell(our_value, low, high)
            top_cell = metric_cell(top_value, top_low, top_high)
        rows.append([metric, better, espn_cell, ours_cell, top_cell])
    tex_table(
        GEN_DIR / "espn_comparison_table.tex",
        booktabs(
            ["Metric", "Better", "IBM/ESPN", "All 2024 (95\\% CI)", "Top-500 (95\\% CI)"],
            rows,
            align="llrrr",
        ),
        "Contextual comparison to published IBM/ESPN point-projection metrics. Green indicates the better value. Confidence intervals are player-cluster bootstrap intervals for our 2024 test set; the top-500 slice ranks players by 2024 total PPR points to better match the published top-500+ population.",
        "tab:espn-comparison",
    )

    paired = pd.read_csv(ROOT / "results/improved_vs_original_regression_bootstrap.csv").set_index("metric")
    rmse_row = paired.loc["rmse"]
    mae_row = paired.loc["mae"]
    (GEN_DIR / "paired_improvement_sentence.tex").write_text(
        "A paired player-cluster bootstrap confirms that the average-error gains are robust: "
        f"RMSE reduction is {rmse_row['improvement_estimate']:.2f} "
        f"[{rmse_row['improvement_ci_lower']:.2f}, {rmse_row['improvement_ci_upper']:.2f}], "
        f"and MAE reduction is {mae_row['improvement_estimate']:.2f} "
        f"[{mae_row['improvement_ci_lower']:.2f}, {mae_row['improvement_ci_upper']:.2f}].\n",
        encoding="utf-8",
    )


def ablation_table():
    abl = pd.read_csv(ROOT / "results/regression_ablation_results.csv")
    abl = abl[abl["model"].eq("lightgbm_expected")].copy()
    labels = {
        "rolling_only": "Rolling only",
        "rolling_plus_previous_season": "+ Previous season",
        "full_without_defense": "Full minus defense",
        "full_features": "+ Defensive matchup",
    }
    rows = []
    for key in ["rolling_only", "rolling_plus_previous_season", "full_without_defense", "full_features"]:
        row = abl[abl["feature_set"].eq(key)]
        rows.append(
            [
                labels[key],
                int(row["num_features"].max()),
                fmt(row[row["position"].eq("QB")]["rmse"].iloc[0]),
                fmt(row[row["position"].eq("RB")]["rmse"].iloc[0]),
                fmt(row[row["position"].eq("WRTE")]["rmse"].iloc[0]),
            ]
        )
    tex_table(
        GEN_DIR / "ablation_table.tex",
        booktabs(["Feature set", "Max feats.", "QB RMSE", "RB RMSE", "WR/TE RMSE"], rows, align="lrrrr"),
        "Original LightGBM feature ablation on the 2024 test season.",
        "tab:ablation",
    )


def boom_table():
    base = pd.read_csv(ROOT / "results/boom_classification_baselines.csv")
    imp = pd.read_csv(ROOT / "results/improved_boom_classification_results.csv")
    base_best = base.loc[base.groupby(["position", "threshold"])["f1"].idxmax()]
    imp_best = imp.loc[imp.groupby(["position", "threshold"])["f1"].idxmax()]
    merged = base_best.merge(
        imp_best,
        on=["position", "threshold"],
        suffixes=("_base", "_enh"),
    ).sort_values(["position", "threshold"])
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            [
                row["position"].replace("WRTE", "WR/TE"),
                f"{int(row['threshold'])}+",
                pct(row["positive_rate_actual_base"]),
                fmt(row["f1_base"], 3),
                fmt(row["f1_enh"], 3),
                fmt(row["f1_enh"] - row["f1_base"], 3),
                fmt(row["precision_enh"], 3),
                fmt(row["recall_enh"], 3),
            ]
        )
    tex_table(
        GEN_DIR / "boom_f1_table.tex",
        booktabs(
            ["Pos.", "Thr.", "Rate", "Base F1", "Enh. F1", "$\\Delta$", "Enh. Prec.", "Enh. Rec."],
            rows,
            align="llrrrrrr",
        ),
        "Best boom-classification F1 by position and threshold on the 2024 test season. The enhanced model is the best of balanced logistic regression, balanced LightGBM, and isotonic-calibrated LightGBM after threshold tuning on 2023.",
        "tab:boom",
    )


def plots():
    enh = pd.read_csv(ROOT / "results/improved_regression_predictions_2024.csv")
    boom = pd.read_csv(ROOT / "results/improved_boom_predictions_2024.csv")
    base_cls = pd.read_csv(ROOT / "results/boom_classification_baselines.csv")
    imp_cls = pd.read_csv(ROOT / "results/improved_boom_classification_results.csv")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    ax.set_axis_off()
    boxes = [
        (0.02, 0.55, 0.17, 0.28, "Supabase\nplayer-week\ntables", "#e8f0fe"),
        (0.25, 0.55, 0.18, 0.28, "Leakage-safe\nfeature\nengineering", "#e6f4ea"),
        (0.50, 0.68, 0.13, 0.18, "Train\n2018-2022", "#fff7e6"),
        (0.66, 0.68, 0.13, 0.18, "Validate\n2023", "#fff7e6"),
        (0.82, 0.68, 0.13, 0.18, "Test\n2024", "#fff7e6"),
        (0.50, 0.24, 0.20, 0.22, "Regression\nexpected + p90", "#fce8e6"),
        (0.74, 0.24, 0.21, 0.22, "Boom classifiers\n10+, 15+, 20+", "#f3e8fd"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor="#444444",
            facecolor=color,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2, transform=ax.transAxes)

    arrows = [
        ((0.19, 0.69), (0.25, 0.69)),
        ((0.43, 0.69), (0.50, 0.77)),
        ((0.63, 0.77), (0.66, 0.77)),
        ((0.79, 0.77), (0.82, 0.77)),
        ((0.565, 0.68), (0.60, 0.46)),
        ((0.725, 0.68), (0.835, 0.46)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#555555",
                transform=ax.transAxes,
            )
        )
    ax.text(
        0.02,
        0.16,
        "2017 is used only as prior history; no hyperparameters, thresholds, or calibration choices use 2024.",
        ha="left",
        va="center",
        fontsize=8,
        color="#444444",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pipeline_temporal_split.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "pipeline_temporal_split.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colors = {"QB": "#386cb0", "RB": "#1b9e77", "WRTE": "#d95f02"}
    bins = np.arange(0, 46, 2.5)
    for pos, group in enh.groupby("position_group"):
        axes[0].hist(
            group["fantasy_points_ppr"],
            bins=bins,
            density=True,
            alpha=0.34,
            color=colors[pos],
            label=pos.replace("WRTE", "WR/TE"),
        )
    axes[0].axvline(20, color="#444444", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Actual PPR fantasy points")
    axes[0].set_ylabel("Density")
    axes[0].set_title("2024 score distribution")
    axes[0].legend(frameon=False, fontsize=8)

    rates = (
        boom.groupby(["position_group", "threshold"])["actual_boom"]
        .mean()
        .reset_index()
    )
    positions = ["QB", "RB", "WRTE"]
    thresholds = [10, 15, 20]
    x = np.arange(len(thresholds))
    width = 0.24
    for i, pos in enumerate(positions):
        vals = [
            rates[(rates["position_group"].eq(pos)) & (rates["threshold"].eq(t))]["actual_boom"].iloc[0]
            for t in thresholds
        ]
        axes[1].bar(x + (i - 1) * width, vals, width, color=colors[pos], label=pos.replace("WRTE", "WR/TE"))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{t}+" for t in thresholds])
    axes[1].set_ylim(0, 0.75)
    axes[1].set_ylabel("Positive rate")
    axes[1].set_xlabel("Boom threshold")
    axes[1].set_title("Boom rates by task")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distribution_and_boom_rates.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "distribution_and_boom_rates.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.75),
        gridspec_kw={"width_ratios": [0.92, 1.25], "wspace": 0.30},
    )
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 60]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35+"]
    tmp = enh.copy()
    tmp["score_bin"] = pd.cut(tmp["fantasy_points_ppr"], bins=bins, labels=labels, include_lowest=True, right=False)

    ax = axes[0]
    counts = tmp["score_bin"].value_counts().reindex(labels).fillna(0)
    ax.bar(labels, counts.values, color="#9aa6b2", width=0.78)
    ax.axvline(3.5, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Actual PPR bin")
    ax.set_ylabel("Player-weeks")
    ax.set_title("2024 score distribution")
    ax.tick_params(axis="x", labelrotation=35, labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    for idx in range(4, len(labels)):
        ax.text(idx, counts.values[idx] + max(counts.values) * 0.025, f"{int(counts.values[idx])}", ha="center", va="bottom", fontsize=6.7)

    ax = axes[1]
    for col, label, color, marker in [
        ("enhanced_lightgbm_expected_tuned_error", "Expected model", "#386cb0", "o"),
        ("enhanced_lightgbm_quantile_p90_error", "p90 ceiling model", "#d95f02", "s"),
    ]:
        means = tmp.groupby("score_bin", observed=True)[col].mean().reindex(labels)
        ax.plot(labels, means, marker=marker, color=color, linewidth=1.8, label=label)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axvline(3.5, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Actual PPR bin")
    ax.set_ylabel("Mean error\n(predicted - actual)")
    ax.set_title("Error increases in the tail")
    ax.tick_params(axis="x", labelrotation=35, labelsize=7.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tail_residuals.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "tail_residuals.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    comparisons = [
        ("RMSE", 6.78, 6.0572, "lower is better"),
        ("Within 10 pts", 88.2, 90.67, "higher is better"),
        ("Within 7 pts", 71.0, 80.21, "higher is better"),
    ]
    for ax, (metric, espn, ours, subtitle) in zip(axes, comparisons):
        bars = ax.bar(["IBM/ESPN", "Ours"], [espn, ours], color=["#7a7a7a", "#386cb0"], width=0.62)
        ax.set_title(f"{metric}\n({subtitle})", fontsize=9)
        ylim_top = max(espn, ours) * (1.18 if metric == "RMSE" else 1.08)
        ax.set_ylim(0, ylim_top)
        for bar, value in zip(bars, [espn, ours]):
            label = f"{value:.2f}" if metric == "RMSE" else f"{value:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, value + ylim_top * 0.025, label, ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelrotation=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Contextual comparison to published IBM/ESPN projection metrics", y=1.08, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "espn_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "espn_comparison.png", bbox_inches="tight")
    plt.close(fig)

    base_best = base_cls.loc[base_cls.groupby(["position", "threshold"])["f1"].idxmax()]
    imp_best = imp_cls.loc[imp_cls.groupby(["position", "threshold"])["f1"].idxmax()]
    merged = base_best.merge(imp_best, on=["position", "threshold"], suffixes=("_base", "_enh"))
    merged["delta"] = merged["f1_enh"] - merged["f1_base"]

    fig, ax = plt.subplots(figsize=(4.8, 2.85))
    positions = ["QB", "RB", "WRTE"]
    thresholds = [10, 15, 20]
    f1 = merged.pivot(index="position", columns="threshold", values="f1_enh").loc[positions, thresholds]
    delta = merged.pivot(index="position", columns="threshold", values="delta").loc[positions, thresholds]
    rate = merged.pivot(index="position", columns="threshold", values="positive_rate_actual_base").loc[positions, thresholds]
    im = ax.imshow(f1.values, cmap="YlGnBu", vmin=0.25, vmax=0.90)
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels([f"{t}+" for t in thresholds])
    ax.set_yticks(np.arange(len(positions)))
    ax.set_yticklabels(["QB", "RB", "WR/TE"])
    ax.set_xlabel("Boom threshold", fontsize=8.5)
    ax.set_title("Boom classification: F1, improvement, and event rate", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    for i in range(len(positions)):
        for j in range(len(thresholds)):
            value = f1.values[i, j]
            text_color = "white" if value > 0.62 else "#1f1f1f"
            ax.text(
                j,
                i,
                f"F1 {value:.3f}\n$\\Delta$ {delta.values[i, j]:+.3f}\nrate {100 * rate.values[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=7.2,
                color=text_color,
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.78, pad=0.03)
    cbar.set_label("Enhanced F1", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boom_classification_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "boom_classification_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    calib_colors = {10: "#386cb0", 15: "#1b9e77", 20: "#d95f02"}
    prob_col = "enhanced_lgbm_isotonic_probability"
    for threshold in thresholds:
        task = boom[boom["threshold"].eq(threshold)].copy()
        if task.empty:
            continue
        task = task.sort_values(prob_col).reset_index(drop=True)
        task["bin"] = pd.qcut(task.index, q=6, labels=False, duplicates="drop")
        calibration = (
            task.groupby("bin", observed=True)
            .agg(predicted=(prob_col, "mean"), observed=("actual_boom", "mean"))
            .dropna()
        )
        ax.plot(
            calibration["predicted"],
            calibration["observed"],
            marker="o",
            linewidth=1.5,
            markersize=4,
            color=calib_colors[threshold],
            label=f"{threshold}+",
        )
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=0.8, linestyle="--", label="Ideal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted boom probability", fontsize=8.5)
    ax.set_ylabel("Observed boom rate", fontsize=8.5)
    ax.set_title("Probability calibration", fontsize=9.5)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boom_probability_calibration.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "boom_probability_calibration.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    pivot = merged.pivot(index="position", columns="threshold", values="delta").loc[["QB", "RB", "WRTE"], [10, 15, 20]]
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-0.04, vmax=0.04)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["10+", "15+", "20+"])
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(["QB", "RB", "WR/TE"])
    ax.set_xlabel("Boom threshold")
    ax.set_title("Enhanced classifier F1 change")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("F1 change", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boom_f1_delta.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "boom_f1_delta.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    dataset_summary()
    regression_tables()
    ablation_table()
    boom_table()
    plots()
    print(f"Wrote artifacts under {PAPER}")


if __name__ == "__main__":
    main()
