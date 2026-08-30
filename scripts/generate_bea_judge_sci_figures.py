from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "datasets" / "model_outputs" / "sci_tables_v2_20260521_110114"
OUT_DIR = ROOT / "paper" / "figures_bea_judge_10k"

PALETTE = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "teal": "#72B7B2",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#111827",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.labelcolor": PALETTE["dark"],
            "xtick.color": PALETTE["dark"],
            "ytick.color": PALETTE["dark"],
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def save_all(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / stem
    fig.savefig(f"{base}.svg", bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    fig.savefig(f"{base}.png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(f"{base}.tiff", dpi=600, bbox_inches="tight")
    except OSError:
        tiff_path = Path(f"{base}.tiff")
        if not tiff_path.exists() or tiff_path.stat().st_size == 0:
            raise
    plt.close(fig)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def figure_1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.set_axis_off()
    nodes = [
        ("BEA-Judge-10K v2\nn=10,200\nlicense/provenance gated", 0.05, 0.58, PALETTE["blue"]),
        ("Base Judge\nM-Prometheus-3B\ns_i=[sA,sB,sT]\nm_i=|sA-sB|", 0.25, 0.58, PALETTE["teal"]),
        ("Bias profile\nr_bias=g(pos,len,fmt,\nrubric,reason)", 0.45, 0.58, PALETTE["orange"]),
        ("Evidence profile\nS(u,v), gap_num/date/entity\nr_evid=max(local risks)", 0.65, 0.58, PALETTE["green"]),
        ("Dual-head calibration\np=softmax((Wφ+b)/T)\nTie policy + confidence", 0.85, 0.58, PALETTE["purple"]),
    ]
    box_w, box_h = 0.16, 0.26
    for text, x, y, color in nodes:
        patch = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color + "22",
        )
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5, color=PALETTE["dark"])
    for i in range(len(nodes) - 1):
        x1 = nodes[i][1] + box_w / 2
        x2 = nodes[i + 1][1] - box_w / 2
        y = nodes[i][2]
        ax.add_patch(
            FancyArrowPatch(
                (x1, y),
                (x2, y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.8,
                color=PALETTE["gray"],
            )
        )
    output = FancyBboxPatch(
        (0.29, 0.12),
        0.42,
        0.18,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=PALETTE["dark"],
        facecolor="#F9FAFB",
    )
    ax.add_patch(output)
    ax.text(
        0.5,
        0.21,
        "calibrated_results.json\npredicted_label, confidence=max p, risk=1-conf+lambda_b*r_bias+lambda_e*r_evid,\nreview_flag / review_reason / SCI tables",
        ha="center",
        va="center",
        fontsize=6.8,
        color=PALETTE["dark"],
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.85, 0.45),
            (0.68, 0.30),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.8,
            color=PALETTE["gray"],
            connectionstyle="arc3,rad=-0.2",
        )
    )
    ax.text(0.5, 0.93, "Figure 1. BEA-Judge 四模块公式化流程", ha="center", fontsize=10, fontweight="bold")
    save_all(fig, "fig1_pipeline")


def figure_2_dataset() -> None:
    dist = read_csv("v2_distribution_table.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    task = dist[dist["dimension"] == "by_task_type"]
    split = dist[dist["dimension"] == "by_split"]
    labels = dist[dist["dimension"].isin(["human_label_distribution", "factuality_label_distribution"])]
    datasets = dist[dist["dimension"] == "by_dataset"].sort_values("count", ascending=True)

    axes[0, 0].bar(task["value"], task["count"], color=[PALETTE["green"], PALETTE["blue"], PALETTE["orange"]])
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Task types")
    axes[0, 0].tick_params(axis="x", rotation=20)
    add_panel_label(axes[0, 0], "a")

    axes[0, 1].bar(split["value"], split["count"], color=[PALETTE["blue"], PALETTE["orange"], PALETTE["green"]])
    axes[0, 1].set_title("Train/dev/test")
    add_panel_label(axes[0, 1], "b")

    axes[1, 0].bar(labels["value"], labels["count"], color=PALETTE["teal"])
    axes[1, 0].set_title("Label distribution")
    axes[1, 0].tick_params(axis="x", rotation=35)
    add_panel_label(axes[1, 0], "c")

    axes[1, 1].barh(datasets["value"], datasets["count"], color=PALETTE["purple"])
    axes[1, 1].set_title("Dataset sources")
    add_panel_label(axes[1, 1], "d")

    fig.suptitle("Figure 2. BEA-Judge-10K v2 数据构成", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_all(fig, "fig2_dataset_distribution")


def figure_3_main_results_ci() -> None:
    ci = read_csv("metric_confidence_interval_table.csv")
    ci = ci[(ci["split"] == "test") & (ci["metric"].isin(["accuracy", "macro_f1", "ece", "brier", "tie_recall"]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    for ax, head, color in zip(axes, ["pairwise", "factuality"], [PALETTE["blue"], PALETTE["green"]]):
        sub = ci[ci["head"] == head].dropna(subset=["point"])
        y = np.arange(len(sub))
        xerr = np.vstack([sub["point"] - sub["ci95_low"], sub["ci95_high"] - sub["point"]])
        ax.errorbar(sub["point"], y, xerr=xerr, fmt="o", color=color, ecolor=color, capsize=3, markersize=4)
        ax.set_yticks(y, sub["metric"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Point estimate and 95% CI")
        ax.set_title(f"{head} test")
        ax.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.5)
    add_panel_label(axes[0], "a")
    add_panel_label(axes[1], "b")
    fig.suptitle("Figure 3. 主实验结果与 bootstrap 95% CI", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_all(fig, "fig3_main_results_ci")


def figure_4_ablation_significance() -> None:
    abl = read_csv("ablation_table.csv")
    sig = read_csv("ablation_significance_table.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))

    pair = abl[abl["head"] == "pairwise"]
    axes[0].bar(pair["variant"], pair["macro_f1"], color=PALETTE["blue"])
    axes[0].set_ylim(0.58, 0.72)
    axes[0].set_ylabel("Macro-F1")
    axes[0].set_title("Pairwise ablation")
    axes[0].tick_params(axis="x", rotation=35)
    add_panel_label(axes[0], "a")

    fact = abl[abl["head"] == "factuality"]
    axes[1].bar(fact["variant"], fact["macro_f1"], color=PALETTE["green"])
    axes[1].set_ylim(0.60, 0.78)
    axes[1].set_title("Factuality ablation")
    axes[1].tick_params(axis="x", rotation=35)
    add_panel_label(axes[1], "b")

    tie = pair.dropna(subset=["tie_recall"])
    axes[2].plot(tie["variant"], tie["tie_recall"], marker="o", color=PALETTE["orange"], linewidth=1.4)
    axes[2].set_ylim(0.34, 0.58)
    axes[2].set_ylabel("Tie recall")
    axes[2].set_title("Tie policy effect")
    axes[2].tick_params(axis="x", rotation=35)
    add_panel_label(axes[2], "c")

    p_cal = sig[(sig["variant"] == "w/o Calibration") & (sig["head"] == "pairwise")]["mcnemar_p"].iloc[0]
    p_evi = sig[(sig["variant"] == "w/o Evidence Module") & (sig["head"] == "factuality")]["mcnemar_p"].iloc[0]
    fig.text(0.50, 0.01, f"McNemar: calibration pairwise p={p_cal:.5f}; evidence factuality p={p_evi:.6f}", ha="center", fontsize=7)
    fig.suptitle("Figure 4. 消融结果与显著性证据", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    save_all(fig, "fig4_ablation_significance")


def figure_5_ragtruth() -> None:
    per = read_csv("per_dataset_table.csv")
    rag = read_csv("ragtruth_results_table.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    sub = per[(per["head"] == "factuality") & (per["dataset"].isin(["ares_nq", "ragtruth"]))]
    x = np.arange(len(sub))
    axes[0].bar(x - 0.18, sub["accuracy"], width=0.35, label="Accuracy", color=PALETTE["blue"])
    axes[0].bar(x + 0.18, sub["macro_f1"], width=0.35, label="Macro-F1", color=PALETTE["green"])
    axes[0].set_xticks(x, sub["dataset"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Factuality datasets")
    axes[0].legend(fontsize=6)
    add_panel_label(axes[0], "a")

    errors = rag[rag["split"] == "test"][["supported_to_unsupported", "unsupported_to_supported"]].iloc[0]
    axes[1].bar(["sup→unsup", "unsup→sup"], errors.values, color=[PALETTE["orange"], PALETTE["red"]])
    axes[1].set_ylabel("Error count")
    axes[1].set_title("RAGTruth error types")
    add_panel_label(axes[1], "b")

    axes[2].bar(rag["split"], rag["review_rate"], color=PALETTE["purple"])
    axes[2].set_ylim(0, 0.8)
    axes[2].set_title("Review capture pressure")
    axes[2].set_ylabel("Review rate")
    add_panel_label(axes[2], "c")

    fig.suptitle("Figure 5. RAGTruth response-level hallucination 难点", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_all(fig, "fig5_ragtruth_analysis")


def figure_6_bias_evidence() -> None:
    bias = read_csv("bias_subgroup_table.csv").sort_values("macro_f1")
    ev = read_csv("evidence_feature_group_ablation_table.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    axes[0].barh(bias["bias_group"], bias["macro_f1"], color=PALETTE["orange"])
    axes[0].scatter(bias["review_rate"], bias["bias_group"], color=PALETTE["blue"], label="Review rate", zorder=3)
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Macro-F1 / review rate")
    axes[0].set_title("Bias subgroup calibration")
    axes[0].legend(fontsize=6)
    add_panel_label(axes[0], "a")

    x = np.arange(len(ev))
    axes[1].plot(x, ev["accuracy"], marker="o", color=PALETTE["blue"], label="Accuracy")
    axes[1].plot(x, ev["macro_f1"], marker="o", color=PALETTE["green"], label="Macro-F1")
    axes[1].set_xticks(x, ev["feature_group"], rotation=35, ha="right")
    axes[1].set_ylim(0.64, 0.79)
    axes[1].set_title("Evidence feature groups")
    axes[1].legend(fontsize=6)
    add_panel_label(axes[1], "b")

    fig.suptitle("Figure 6. 偏差复核与证据特征诊断", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_all(fig, "fig6_bias_evidence_diagnostics")


def write_manifest() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("fig*.*")):
        rows.append({"file": path.name, "bytes": path.stat().st_size})
    with (OUT_DIR / "figure_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "bytes"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_1_pipeline()
    figure_2_dataset()
    figure_3_main_results_ci()
    figure_4_ablation_significance()
    figure_5_ragtruth()
    figure_6_bias_evidence()
    write_manifest()
    print(f"figures={OUT_DIR}")


if __name__ == "__main__":
    main()
