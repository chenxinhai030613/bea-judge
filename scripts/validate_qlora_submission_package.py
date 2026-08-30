"""Validate the submission-ready QLoRA dual-operating-point result package."""

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


def mean_metric(item: Mapping[str, Any], metric: str) -> Optional[float]:
    value = item.get("metrics", {}).get(metric, {}).get("mean")
    return None if value is None else float(value)


def baseline_metric(summary: Mapping[str, Any], metric: str) -> Optional[float]:
    value = summary.get("baseline", {}).get("metrics", {}).get(metric, {}).get("mean")
    return None if value is None else float(value)


def require_metric(value: Optional[float], name: str) -> float:
    if value is None:
        raise ValueError(f"missing metric: {name}")
    return value


def validate_submission_summary(summary: Mapping[str, Any]) -> Dict[str, Any]:
    points = summary.get("operating_points", {})
    accuracy_point = points.get("accuracy_oriented")
    tie_point = points.get("tie_sensitive_dev_selected")
    if not accuracy_point or not tie_point:
        raise ValueError("submission summary must contain both operating points")

    baseline = {metric: require_metric(baseline_metric(summary, metric), f"baseline.{metric}") for metric in METRICS}
    accuracy_metrics = {
        metric: require_metric(mean_metric(accuracy_point, metric), f"accuracy_oriented.{metric}")
        for metric in METRICS
    }
    tie_metrics = {
        metric: require_metric(mean_metric(tie_point, metric), f"tie_sensitive_dev_selected.{metric}")
        for metric in METRICS
    }

    checks = {
        "accuracy_point_macro_f1_above_baseline": accuracy_metrics["macro_f1"] > baseline["macro_f1"],
        "accuracy_point_accuracy_above_baseline": accuracy_metrics["accuracy"] > baseline["accuracy"],
        "accuracy_point_ece_below_baseline": accuracy_metrics["ece"] < baseline["ece"],
        "accuracy_point_gate_expected_not_all_passed": not bool(accuracy_point.get("gate", {}).get("all_passed")),
        "tie_point_gate_all_passed": bool(tie_point.get("gate", {}).get("all_passed")),
        "tie_point_macro_f1_above_baseline": tie_metrics["macro_f1"] > baseline["macro_f1"],
        "tie_point_tie_recall_above_baseline": tie_metrics["tie_recall"] > baseline["tie_recall"],
        "tie_point_ece_below_baseline": tie_metrics["ece"] < baseline["ece"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failed,
        "checks": checks,
        "failed": failed,
        "baseline": baseline,
        "accuracy_oriented": accuracy_metrics,
        "tie_sensitive_dev_selected": tie_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate QLoRA submission-ready result package.")
    parser.add_argument(
        "--submission-summary",
        default="datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_submission_summary(load_json(resolve_root_path(args.submission_summary)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
