"""Build submission-ready QLoRA result summaries from three-seed reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
METRICS = ("accuracy", "macro_f1", "ece", "tie_recall")


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_summary(summary: Mapping[str, Any], system: str) -> Dict[str, Dict[str, Optional[float]]]:
    by_system = summary.get("by_system", {})
    if system not in by_system:
        raise KeyError(f"system not found in summary: {system}")
    return {metric: dict(by_system[system][metric]) for metric in METRICS}


def mean_value(metrics: Mapping[str, Mapping[str, Optional[float]]], metric: str) -> Optional[float]:
    value = metrics.get(metric, {}).get("mean")
    return None if value is None else float(value)


def delta(
    candidate: Mapping[str, Mapping[str, Optional[float]]],
    baseline: Mapping[str, Mapping[str, Optional[float]]],
    metric: str,
) -> Optional[float]:
    left = mean_value(candidate, metric)
    right = mean_value(baseline, metric)
    if left is None or right is None:
        return None
    return round(left - right, 6)


def format_mean_std(item: Mapping[str, Optional[float]]) -> str:
    if item.get("mean") is None:
        return ""
    return f"{float(item['mean']):.4f} +/- {float(item['std'] or 0.0):.4f}"


def build_submission_summary(
    *,
    conservative_summary: Mapping[str, Any],
    tie_sensitive_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline = metric_summary(conservative_summary, "Current BEA-Judge")
    conservative = metric_summary(conservative_summary, "QLoRA-BEA-Judge")
    tie_sensitive = metric_summary(tie_sensitive_summary, "QLoRA-BEA-Judge")
    return {
        "baseline": {
            "system": "Current BEA-Judge",
            "metrics": baseline,
        },
        "operating_points": {
            "accuracy_oriented": {
                "system": "QLoRA-BEA-Judge",
                "source": "1024 three-seed conservative calibrated operating point",
                "gate": conservative_summary.get("gate", {}),
                "metrics": conservative,
                "delta_vs_current": {metric: delta(conservative, baseline, metric) for metric in METRICS},
                "recommended_use": "main accuracy/calibration result",
            },
            "tie_sensitive_dev_selected": {
                "system": "QLoRA-BEA-Judge",
                "source": "1024 three-seed dev-selected helpsteer2 tie-sensitive operating point",
                "gate": tie_sensitive_summary.get("gate", {}),
                "metrics": tie_sensitive,
                "delta_vs_current": {metric: delta(tie_sensitive, baseline, metric) for metric in METRICS},
                "recommended_use": "secondary operating point for tie-sensitive evaluation",
            },
        },
        "recommended_claims": [
            (
                "Under the stable 1024-token three-seed protocol, the accuracy-oriented "
                "QLoRA-BEA-Judge operating point improves macro-F1 and ECE over Current BEA-Judge, "
                "but its Tie recall is lower than the frozen baseline."
            ),
            (
                "A dev-selected tie-sensitive operating point passes all claim-gate checks across "
                "three seeds and substantially improves Tie recall, with a clear accuracy trade-off."
            ),
        ],
    }


def markdown_summary(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline"]["metrics"]
    points = summary["operating_points"]
    lines = [
        "# QLoRA Submission-Ready Result Summary",
        "",
        "This file separates the stable conservative operating point from the dev-selected tie-sensitive operating point.",
        "",
        "## Main Table",
        "",
        "| system / operating point | gate | accuracy | macro_f1 | ece | tie_recall |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        "| Current BEA-Judge | baseline | "
        + " | ".join(format_mean_std(baseline[metric]) for metric in METRICS)
        + " |",
    ]
    for name, item in points.items():
        gate = item.get("gate", {})
        gate_text = f"{gate.get('passed_count', 0)}/{gate.get('total', 0)}"
        metrics = item["metrics"]
        lines.append(
            f"| QLoRA-BEA-Judge ({name}) | {gate_text} | "
            + " | ".join(format_mean_std(metrics[metric]) for metric in METRICS)
            + " |"
        )

    lines.extend(["", "## Delta Vs Current BEA-Judge", ""])
    lines.extend(
        [
            "| operating point | accuracy | macro_f1 | ece | tie_recall |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, item in points.items():
        deltas = item["delta_vs_current"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{float(deltas[metric]):+.4f}" for metric in METRICS)
            + " |"
        )

    lines.extend(["", "## Recommended Claims", ""])
    for claim in summary["recommended_claims"]:
        lines.append(f"- {claim}")
    lines.extend(
        [
            "",
            "## Artifact Index",
            "",
            "- Conservative summary: `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/three_seed_summary.md`",
            "- Tie-sensitive summary: `datasets/model_outputs/qlora_3seed_epoch1_1024_tie_sensitive_dev_summary/three_seed_summary.md`",
            "- System audit: `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/model_system_audit.md`",
            "- Dev-selected Tie audit: `datasets/model_outputs/qlora_3seed_epoch1_1024_summary/tie_sensitive_dev_selected_policy_audit.md`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build submission-ready QLoRA result summary.")
    parser.add_argument(
        "--conservative-summary",
        default="datasets/model_outputs/qlora_3seed_epoch1_1024_summary/three_seed_summary.json",
    )
    parser.add_argument(
        "--tie-sensitive-summary",
        default="datasets/model_outputs/qlora_3seed_epoch1_1024_tie_sensitive_dev_summary/three_seed_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/model_outputs/qlora_3seed_epoch1_1024_summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_submission_summary(
        conservative_summary=load_json(resolve_root_path(args.conservative_summary)),
        tie_sensitive_summary=load_json(resolve_root_path(args.tie_sensitive_summary)),
    )
    output_dir = resolve_root_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qlora_submission_ready_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "qlora_submission_ready_results.md").write_text(
        markdown_summary(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary["operating_points"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
