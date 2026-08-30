"""Build validation-report copies with a dev-selected tie-sensitive policy applied."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tie_sensitive_policy_audit import select_policy  # noqa: E402


PAIRWISE_LABELS = ("A>B", "B>A", "Tie")


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def final_score_from_label(label: str) -> float:
    return {"A>B": 1.0, "Tie": 0.5, "B>A": 0.0}.get(label, 0.5)


def adjusted_prediction(row: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    probs = row.get("label_probabilities", {}) or {}
    pred = str(row.get("predicted_label"))
    if (
        str(row.get("dataset")) == str(policy["dataset"])
        and float(probs.get("Tie", 0.0)) >= float(policy["min_tie_probability"])
        and abs(float(probs.get("A>B", 0.0)) - float(probs.get("B>A", 0.0)))
        <= float(policy["max_ab_margin"])
    ):
        return "Tie"
    return pred


def adjusted_rows(rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        pred = adjusted_prediction(row, policy)
        probs = item.get("label_probabilities", {}) or {}
        item["predicted_label"] = pred
        item["pairwise_label"] = pred
        item["final_score"] = round(final_score_from_label(pred), 4)
        item["confidence"] = round(float(probs.get(pred, item.get("confidence", 0.0))), 6)
        item["risk_score"] = round(1.0 - float(item["confidence"]), 6)
        out.append(item)
    return out


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
        selected = [
            i
            for i, confidence in enumerate(confidences)
            if confidence >= lo and (confidence < hi if index < bins - 1 else confidence <= hi)
        ]
        if not selected:
            continue
        accuracy = sum(y_true[i] == y_pred[i] for i in selected) / len(selected)
        avg_confidence = sum(confidences[i] for i in selected) / len(selected)
        ece += len(selected) / total * abs(accuracy - avg_confidence)
    return float(ece)


def brier_score(rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        probs = row.get("label_probabilities", {}) or {}
        gold = str(row.get("human_label"))
        total += sum((float(probs.get(label, 0.0)) - (1.0 if gold == label else 0.0)) ** 2 for label in PAIRWISE_LABELS)
    return total / len(rows)


def confusion(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {label: {pred: 0 for pred in PAIRWISE_LABELS} for label in PAIRWISE_LABELS}
    for actual, pred in zip(y_true, y_pred):
        if actual in out and pred in out[actual]:
            out[actual][pred] += 1
    return out


def metrics_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    y_true = [str(row.get("human_label")) for row in rows if row.get("human_label") in PAIRWISE_LABELS]
    y_pred = [str(row.get("predicted_label")) for row in rows if row.get("human_label") in PAIRWISE_LABELS]
    confidences = [
        float((row.get("label_probabilities", {}) or {}).get(str(row.get("predicted_label")), row.get("confidence", 0.0)))
        for row in rows
        if row.get("human_label") in PAIRWISE_LABELS
    ]
    tie_indices = [i for i, label in enumerate(y_true) if label == "Tie"]
    return {
        "accuracy": round(sum(actual == pred for actual, pred in zip(y_true, y_pred)) / len(y_true), 4)
        if y_true
        else 0.0,
        "macro_f1": round(macro_f1(y_true, y_pred), 4) if y_true else 0.0,
        "ece": round(ece_score(y_true, y_pred, confidences), 4) if y_true else 0.0,
        "brier": round(brier_score(rows), 4),
        "tie_recall": round(sum(y_pred[i] == "Tie" for i in tie_indices) / len(tie_indices), 4)
        if tie_indices
        else None,
        "confusion": confusion(y_true, y_pred),
        "pred_distribution": dict(Counter(y_pred)),
        "gold_distribution": dict(Counter(y_true)),
    }


def build_seed_report(
    *,
    seed: str,
    validation_report: Path,
    calibrated_results: Path,
    frozen_dev: Mapping[str, float],
    args: argparse.Namespace,
) -> Tuple[Path, Dict[str, Any]]:
    report = load_json(validation_report)
    calibrated = load_json(calibrated_results)
    policy, candidates = select_policy(
        calibrated["dev"]["pairwise"],
        dataset=args.dataset,
        frozen_dev=frozen_dev,
        macro_gain_min=float(args.macro_gain_min),
        ece_max=float(args.ece_max),
        thresholds=[float(value) for value in args.thresholds],
        margins=[float(value) for value in args.margins],
    )
    if policy is None:
        raise RuntimeError(f"no eligible tie-sensitive policy for seed {seed}")

    adjusted_calibrated = deepcopy(calibrated)
    for split in ("train", "dev", "test"):
        if split in adjusted_calibrated and "pairwise" in adjusted_calibrated[split]:
            adjusted_calibrated[split]["pairwise"] = adjusted_rows(adjusted_calibrated[split]["pairwise"], policy)

    adjusted_report = deepcopy(report)
    adjusted_report.setdefault("tie_sensitive_policy", {})
    adjusted_report["tie_sensitive_policy"] = {
        "status": "dev_selected_policy_applied",
        "seed": seed,
        "policy": policy,
        "selection_constraints": {
            "dataset": args.dataset,
            "accuracy_min": float(frozen_dev["accuracy"]),
            "macro_f1_min": round(float(frozen_dev["macro_f1"]) + float(args.macro_gain_min), 6),
            "ece_max": float(args.ece_max),
        },
        "candidate_count": len(candidates),
        "eligible_count": sum(1 for row in candidates if row["eligible"]),
    }
    adjusted_report["heads"]["pairwise"]["calibrated_dev_metrics"] = metrics_for_rows(
        adjusted_calibrated["dev"]["pairwise"]
    )
    adjusted_report["test_evaluation"]["pairwise"]["metrics"] = metrics_for_rows(
        adjusted_calibrated["test"]["pairwise"]
    )
    adjusted_report["train_evaluation"]["pairwise"]["metrics"] = metrics_for_rows(
        adjusted_calibrated["train"]["pairwise"]
    )

    run_dir = resolve_root_path(args.output_template.format(seed=seed))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "validation_report.json").write_text(
        json.dumps(adjusted_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "calibrated_results.json").write_text(
        json.dumps(adjusted_calibrated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "tie_sensitive_policy.json").write_text(
        json.dumps(adjusted_report["tie_sensitive_policy"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir, adjusted_report["tie_sensitive_policy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build tie-sensitive validation report copies.")
    parser.add_argument("--seeds", nargs="*", default=["13", "42", "2026"])
    parser.add_argument(
        "--validation-template",
        default="datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024/validation_report.json",
    )
    parser.add_argument(
        "--calibrated-template",
        default="datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024/calibrated_results.json",
    )
    parser.add_argument(
        "--output-template",
        default="datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024_tie_sensitive_dev",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen = load_json(resolve_root_path(args.frozen_report))
    frozen_dev = dict(frozen["heads"]["pairwise"]["calibrated_dev_metrics"])
    outputs = {}
    for seed in args.seeds:
        run_dir, policy = build_seed_report(
            seed=str(seed),
            validation_report=resolve_root_path(args.validation_template.format(seed=seed)),
            calibrated_results=resolve_root_path(args.calibrated_template.format(seed=seed)),
            frozen_dev=frozen_dev,
            args=args,
        )
        outputs[str(seed)] = {"run_dir": str(run_dir), "policy": policy}
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
