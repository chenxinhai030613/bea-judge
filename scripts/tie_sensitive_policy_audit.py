"""Audit a dev-selected tie-sensitive operating point for QLoRA-BEA outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
PAIRWISE_LABELS = ("A>B", "B>A", "Tie")


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    scores: List[float] = []
    for label in PAIRWISE_LABELS:
        tp = sum(actual == label and pred == label for actual, pred in zip(y_true, y_pred))
        fp = sum(actual != label and pred == label for actual, pred in zip(y_true, y_pred))
        fn = sum(actual == label and pred != label for actual, pred in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def ece_score(y_true: Sequence[str], y_pred: Sequence[str], confidences: Sequence[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        mask = [
            i
            for i, confidence in enumerate(confidences)
            if confidence >= lo and (confidence < hi if index < bins - 1 else confidence <= hi)
        ]
        if not mask:
            continue
        accuracy = sum(y_true[i] == y_pred[i] for i in mask) / len(mask)
        avg_confidence = sum(confidences[i] for i in mask) / len(mask)
        ece += len(mask) / total * abs(accuracy - avg_confidence)
    return float(ece)


def apply_policy_to_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    min_tie_probability: float,
    max_ab_margin: float,
) -> Dict[str, Any]:
    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []
    for row in rows:
        probs = row.get("label_probabilities", {}) or {}
        pred = str(row.get("predicted_label"))
        if (
            str(row.get("dataset")) == dataset
            and float(probs.get("Tie", 0.0)) >= min_tie_probability
            and abs(float(probs.get("A>B", 0.0)) - float(probs.get("B>A", 0.0))) <= max_ab_margin
        ):
            pred = "Tie"
        y_true.append(str(row.get("human_label")))
        y_pred.append(pred)
        confidences.append(float(probs.get(pred, 0.0)))

    tie_indices = [i for i, label in enumerate(y_true) if label == "Tie"]
    return {
        "accuracy": round(sum(actual == pred for actual, pred in zip(y_true, y_pred)) / len(y_true), 6)
        if y_true
        else 0.0,
        "macro_f1": round(macro_f1(y_true, y_pred), 6) if y_true else 0.0,
        "ece": round(ece_score(y_true, y_pred, confidences), 6) if y_true else 0.0,
        "tie_recall": round(
            sum(y_pred[i] == "Tie" for i in tie_indices) / len(tie_indices),
            6,
        )
        if tie_indices
        else None,
        "tie_pred_count": sum(label == "Tie" for label in y_pred),
    }


def frozen_dev_metrics(path: Path) -> Dict[str, float]:
    report = load_json(path)
    return dict(report["heads"]["pairwise"]["calibrated_dev_metrics"])


def select_policy(
    dev_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    frozen_dev: Mapping[str, float],
    macro_gain_min: float,
    ece_max: float,
    thresholds: Sequence[float],
    margins: Sequence[float],
) -> Tuple[Optional[Dict[str, float]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for threshold in thresholds:
        for margin in margins:
            metrics = apply_policy_to_rows(
                dev_rows,
                dataset=dataset,
                min_tie_probability=threshold,
                max_ab_margin=margin,
            )
            row = {
                "dataset": dataset,
                "min_tie_probability": threshold,
                "max_ab_margin": margin,
                "metrics": metrics,
                "eligible": (
                    metrics["accuracy"] >= float(frozen_dev["accuracy"])
                    and metrics["macro_f1"] >= float(frozen_dev["macro_f1"]) + macro_gain_min
                    and metrics["ece"] <= ece_max
                    and metrics["tie_recall"] is not None
                ),
            }
            candidates.append(row)
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return None, candidates
    best = max(
        eligible,
        key=lambda row: (
            float(row["metrics"].get("tie_recall") or 0.0),
            float(row["metrics"]["macro_f1"]),
            float(row["metrics"]["accuracy"]),
            -float(row["metrics"]["ece"]),
        ),
    )
    policy = {
        "dataset": dataset,
        "min_tie_probability": float(best["min_tie_probability"]),
        "max_ab_margin": float(best["max_ab_margin"]),
    }
    return policy, candidates


def summarize_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Optional[float]]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": round(float(statistics.fmean(values)), 6),
        "std": round(float(statistics.stdev(values)), 6) if len(values) > 1 else 0.0,
    }


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    frozen_dev = frozen_dev_metrics(resolve_root_path(args.frozen_report))
    thresholds = [float(value) for value in args.thresholds]
    margins = [float(value) for value in args.margins]
    per_seed: Dict[str, Any] = {}
    for seed in args.seeds:
        path = resolve_root_path(args.calibrated_template.format(seed=seed))
        payload = load_json(path)
        dev_rows = payload["dev"]["pairwise"]
        test_rows = payload["test"]["pairwise"]
        policy, candidates = select_policy(
            dev_rows,
            dataset=args.dataset,
            frozen_dev=frozen_dev,
            macro_gain_min=float(args.macro_gain_min),
            ece_max=float(args.ece_max),
            thresholds=thresholds,
            margins=margins,
        )
        if policy is None:
            per_seed[str(seed)] = {
                "selected_policy": None,
                "dev_metrics": None,
                "test_metrics": None,
                "eligible_count": 0,
                "candidate_count": len(candidates),
            }
            continue
        dev_metrics = apply_policy_to_rows(dev_rows, **policy)
        test_metrics = apply_policy_to_rows(test_rows, **policy)
        per_seed[str(seed)] = {
            "selected_policy": policy,
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics,
            "eligible_count": sum(1 for row in candidates if row["eligible"]),
            "candidate_count": len(candidates),
        }

    mean_std: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for split in ("dev", "test"):
        metrics_rows = [
            row[f"{split}_metrics"]
            for row in per_seed.values()
            if row.get(f"{split}_metrics") is not None
        ]
        mean_std[split] = {
            metric: summarize_metric(metrics_rows, metric)
            for metric in ("accuracy", "macro_f1", "ece", "tie_recall")
        }
    return {
        "status": "dev_selected_tie_sensitive_policy_audit",
        "selection_constraints": {
            "dataset": args.dataset,
            "accuracy_min": frozen_dev["accuracy"],
            "macro_f1_min": round(float(frozen_dev["macro_f1"]) + float(args.macro_gain_min), 6),
            "ece_max": float(args.ece_max),
            "selection_order": "maximize dev tie_recall, then macro_f1, accuracy, -ece",
        },
        "frozen_dev_metrics": frozen_dev,
        "seeds": [str(seed) for seed in args.seeds],
        "per_seed": per_seed,
        "mean_std": mean_std,
    }


def format_mean_std(item: Mapping[str, Optional[float]]) -> str:
    if item.get("mean") is None:
        return ""
    return f"{item['mean']:.4f} +/- {item['std']:.4f}"


def markdown_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Dev-Selected Tie-Sensitive Policy Audit",
        "",
        "This audit selects the operating point on dev only and applies it once to test.",
        "",
        f"Selection constraints: `{audit['selection_constraints']}`",
        "",
        "| split | accuracy | macro_f1 | ece | tie_recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for split in ("dev", "test"):
        row = audit["mean_std"][split]
        lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    format_mean_std(row["accuracy"]),
                    format_mean_std(row["macro_f1"]),
                    format_mean_std(row["ece"]),
                    format_mean_std(row["tie_recall"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per Seed",
            "",
            "| seed | threshold | margin | dev_accuracy | dev_macro_f1 | dev_ece | dev_tie_recall | test_accuracy | test_macro_f1 | test_ece | test_tie_recall |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for seed, row in audit["per_seed"].items():
        policy = row.get("selected_policy") or {}
        dev = row.get("dev_metrics") or {}
        test = row.get("test_metrics") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    seed,
                    str(policy.get("min_tie_probability", "")),
                    str(policy.get("max_ab_margin", "")),
                    str(dev.get("accuracy", "")),
                    str(dev.get("macro_f1", "")),
                    str(dev.get("ece", "")),
                    str(dev.get("tie_recall", "")),
                    str(test.get("accuracy", "")),
                    str(test.get("macro_f1", "")),
                    str(test.get("ece", "")),
                    str(test.get("tie_recall", "")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dev-selected tie-sensitive operating points.")
    parser.add_argument("--seeds", nargs="*", default=["13", "42", "2026"])
    parser.add_argument(
        "--calibrated-template",
        default="datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024/calibrated_results.json",
    )
    parser.add_argument(
        "--frozen-report",
        default="datasets/model_outputs/bea_judge_20260521_110114/validation_report.json",
    )
    parser.add_argument("--dataset", default="helpsteer2")
    parser.add_argument("--macro-gain-min", type=float, default=0.02)
    parser.add_argument("--ece-max", type=float, default=0.06)
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45],
    )
    parser.add_argument(
        "--margins",
        nargs="*",
        type=float,
        default=[0.65, 0.70, 0.75, 0.80, 0.90, 1.0],
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/model_outputs/qlora_3seed_epoch1_1024_summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args)
    output_dir = resolve_root_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tie_sensitive_dev_selected_policy_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "tie_sensitive_dev_selected_policy_audit.md").write_text(
        markdown_report(audit),
        encoding="utf-8",
    )
    print(json.dumps(audit["mean_std"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
