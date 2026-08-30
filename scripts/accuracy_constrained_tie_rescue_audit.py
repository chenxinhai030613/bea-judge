"""Audit accuracy-constrained Tie rescue policies for QLoRA-BEA outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
PAIRWISE_LABELS = ("A>B", "B>A", "Tie")
METRICS = ("accuracy", "macro_f1", "ece", "tie_recall")


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


def rescued_label(row: Mapping[str, Any], policy: Mapping[str, Any]) -> Tuple[str, bool]:
    probs = row.get("label_probabilities", {}) or {}
    pred = str(row.get("predicted_label"))
    if pred not in ("A>B", "B>A"):
        return pred, False
    if str(row.get("dataset")) != str(policy["dataset"]):
        return pred, False

    prob_a = float(probs.get("A>B", 0.0))
    prob_b = float(probs.get("B>A", 0.0))
    prob_tie = float(probs.get("Tie", 0.0))
    should_rescue = (
        prob_tie >= float(policy["min_tie_probability"])
        and abs(prob_a - prob_b) <= float(policy["max_ab_margin"])
        and max(prob_a, prob_b) <= float(policy["max_ab_confidence"])
    )
    return ("Tie", True) if should_rescue else (pred, False)


def metrics_for_rows(rows: Sequence[Mapping[str, Any]], policy: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    y_true: List[str] = []
    y_pred: List[str] = []
    confidences: List[float] = []
    rescued_total = 0
    rescued_correct = 0
    rescued_gold = Counter()
    rescued_original = Counter()

    for row in rows:
        gold = str(row.get("human_label"))
        if gold not in PAIRWISE_LABELS:
            continue
        probs = row.get("label_probabilities", {}) or {}
        original_pred = str(row.get("predicted_label"))
        pred = original_pred
        rescued = False
        if policy is not None:
            pred, rescued = rescued_label(row, policy)
        if rescued:
            rescued_total += 1
            rescued_correct += int(gold == "Tie")
            rescued_gold[gold] += 1
            rescued_original[original_pred] += 1
        y_true.append(gold)
        y_pred.append(pred)
        confidences.append(float(probs.get(pred, row.get("confidence", 0.0))))

    tie_indices = [i for i, label in enumerate(y_true) if label == "Tie"]
    accuracy = sum(actual == pred for actual, pred in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return {
        "accuracy": round(float(accuracy), 6),
        "macro_f1": round(macro_f1(y_true, y_pred), 6) if y_true else 0.0,
        "ece": round(ece_score(y_true, y_pred, confidences), 6) if y_true else 0.0,
        "tie_recall": round(sum(y_pred[i] == "Tie" for i in tie_indices) / len(tie_indices), 6)
        if tie_indices
        else None,
        "tie_pred_count": sum(label == "Tie" for label in y_pred),
        "rescued_count": rescued_total,
        "rescued_precision": round(rescued_correct / rescued_total, 6) if rescued_total else None,
        "rescued_gold_distribution": dict(rescued_gold),
        "rescued_original_distribution": dict(rescued_original),
        "pred_distribution": dict(Counter(y_pred)),
        "gold_distribution": dict(Counter(y_true)),
    }


def guardrail_metrics(path: Path) -> Dict[str, float]:
    report = load_json(path)
    return dict(report["heads"]["pairwise"]["calibrated_dev_metrics"])


def select_policy(
    dev_rows: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    guardrail: Mapping[str, float],
    ece_max: float,
    thresholds: Sequence[float],
    margins: Sequence[float],
    max_ab_confidences: Sequence[float],
    guardrail_epsilon: float,
) -> Tuple[Optional[Dict[str, float]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    for threshold in thresholds:
        for margin in margins:
            for max_ab_confidence in max_ab_confidences:
                policy = {
                    "dataset": dataset,
                    "min_tie_probability": float(threshold),
                    "max_ab_margin": float(margin),
                    "max_ab_confidence": float(max_ab_confidence),
                }
                metrics = metrics_for_rows(dev_rows, policy)
                eligible = (
                    metrics["accuracy"] + guardrail_epsilon >= float(guardrail["accuracy"])
                    and metrics["macro_f1"] + guardrail_epsilon >= float(guardrail["macro_f1"])
                    and metrics["ece"] <= ece_max
                    and metrics["tie_recall"] is not None
                )
                candidates.append({"policy": policy, "metrics": metrics, "eligible": eligible})

    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        return None, candidates

    best = max(
        eligible,
        key=lambda row: (
            float(row["metrics"].get("tie_recall") or 0.0),
            float(row["metrics"]["accuracy"]),
            -float(row["metrics"]["ece"]),
            float(row["metrics"]["macro_f1"]),
            -float(row["metrics"]["rescued_count"]),
        ),
    )
    return dict(best["policy"]), candidates


def selected_or_noop_metrics(
    rows: Sequence[Mapping[str, Any]],
    policy: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return metrics_for_rows(rows, policy) if policy is not None else metrics_for_rows(rows)


def summarize_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Optional[float]]:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": round(float(statistics.fmean(values)), 6),
        "std": round(float(statistics.stdev(values)), 6) if len(values) > 1 else 0.0,
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    return {metric: summarize_metric(rows, metric) for metric in METRICS}


def pass_fail(mean_std: Mapping[str, Mapping[str, Optional[float]]], args: argparse.Namespace) -> Dict[str, bool]:
    return {
        "accuracy_ge_min": float(mean_std["accuracy"]["mean"] or 0.0) >= float(args.test_accuracy_min),
        "tie_recall_gt_min": float(mean_std["tie_recall"]["mean"] or 0.0) > float(args.test_tie_recall_min),
        "macro_f1_ge_min": float(mean_std["macro_f1"]["mean"] or 0.0) >= float(args.test_macro_f1_min),
        "ece_le_max": float(mean_std["ece"]["mean"] or 1.0) <= float(args.test_ece_max),
    }


def build_setting_audit(setting: str, args: argparse.Namespace) -> Dict[str, Any]:
    per_seed: Dict[str, Any] = {}
    for seed in args.seeds:
        calibrated_path = resolve_root_path(args.calibrated_template.format(seed=seed, setting=setting))
        guardrail_path = resolve_root_path(args.guardrail_validation_template.format(seed=seed, setting=setting))
        payload = load_json(calibrated_path)
        guardrail = guardrail_metrics(guardrail_path)
        dev_rows = payload["dev"]["pairwise"]
        test_rows = payload["test"]["pairwise"]
        initial_dev = metrics_for_rows(dev_rows)
        initial_test = metrics_for_rows(test_rows)
        policy, candidates = select_policy(
            dev_rows,
            dataset=args.dataset,
            guardrail=guardrail,
            ece_max=float(args.ece_max),
            thresholds=[float(value) for value in args.thresholds],
            margins=[float(value) for value in args.margins],
            max_ab_confidences=[float(value) for value in args.max_ab_confidences],
            guardrail_epsilon=float(args.guardrail_epsilon),
        )
        if policy is None:
            per_seed[str(seed)] = {
                "selected_policy": None,
                "guardrail_dev_metrics": guardrail,
                "initial_dev_metrics": initial_dev,
                "initial_test_metrics": initial_test,
                "dev_metrics": initial_dev,
                "test_metrics": initial_test,
                "eligible_count": 0,
                "candidate_count": len(candidates),
                "used_noop_fallback": True,
            }
            continue

        per_seed[str(seed)] = {
            "selected_policy": policy,
            "guardrail_dev_metrics": guardrail,
            "initial_dev_metrics": initial_dev,
            "initial_test_metrics": initial_test,
            "dev_metrics": metrics_for_rows(dev_rows, policy),
            "test_metrics": metrics_for_rows(test_rows, policy),
            "eligible_count": sum(1 for row in candidates if row["eligible"]),
            "candidate_count": len(candidates),
            "used_noop_fallback": False,
        }

    mean_std: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for key in ("initial_dev", "initial_test", "dev", "test"):
        rows = [row[f"{key}_metrics"] for row in per_seed.values() if row.get(f"{key}_metrics") is not None]
        mean_std[key] = summarize_rows(rows)

    checks = pass_fail(mean_std["test"], args)
    return {
        "setting": setting,
        "seeds": [str(seed) for seed in args.seeds],
        "selection_constraints": {
            "dataset": args.dataset,
            "guardrail_source": args.guardrail_validation_template,
            "dev_accuracy_min": "per-seed guardrail calibrated_dev_metrics.accuracy",
            "dev_macro_f1_min": "per-seed guardrail calibrated_dev_metrics.macro_f1",
            "ece_max": float(args.ece_max),
            "selection_order": "maximize dev tie_recall, then accuracy, -ece, macro_f1",
        },
        "test_success_criteria": {
            "accuracy_min": float(args.test_accuracy_min),
            "tie_recall_strict_min": float(args.test_tie_recall_min),
            "macro_f1_min": float(args.test_macro_f1_min),
            "ece_max": float(args.test_ece_max),
        },
        "per_seed": per_seed,
        "mean_std": mean_std,
        "success_checks": checks,
        "passed": all(checks.values()),
    }


def mean_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = [float(row[metric]) for row in rows if row.get(metric) is not None]
    return float(statistics.fmean(values)) if values else 0.0


def build_global_setting_audit(setting: str, args: argparse.Namespace) -> Dict[str, Any]:
    seed_payloads: Dict[str, Dict[str, Any]] = {}
    for seed in args.seeds:
        calibrated_path = resolve_root_path(args.calibrated_template.format(seed=seed, setting=setting))
        payload = load_json(calibrated_path)
        seed_payloads[str(seed)] = {
            "dev_rows": payload["dev"]["pairwise"],
            "test_rows": payload["test"]["pairwise"],
        }

    candidates: List[Dict[str, Any]] = []
    for threshold in [float(value) for value in args.thresholds]:
        for margin in [float(value) for value in args.margins]:
            for max_ab_confidence in [float(value) for value in args.max_ab_confidences]:
                policy = {
                    "dataset": args.dataset,
                    "min_tie_probability": threshold,
                    "max_ab_margin": margin,
                    "max_ab_confidence": max_ab_confidence,
                }
                dev_rows = [metrics_for_rows(item["dev_rows"], policy) for item in seed_payloads.values()]
                test_rows = [metrics_for_rows(item["test_rows"], policy) for item in seed_payloads.values()]
                dev_mean = {metric: mean_metric(dev_rows, metric) for metric in METRICS}
                test_mean = {metric: mean_metric(test_rows, metric) for metric in METRICS}
                eligible = (
                    dev_mean["accuracy"] + float(args.guardrail_epsilon) >= float(args.dev_accuracy_min)
                    and dev_mean["macro_f1"] + float(args.guardrail_epsilon) >= float(args.dev_macro_f1_min)
                    and dev_mean["ece"] <= float(args.ece_max)
                    and dev_mean["tie_recall"] > float(args.dev_tie_recall_min)
                )
                candidates.append(
                    {
                        "policy": policy,
                        "dev_mean": dev_mean,
                        "test_mean": test_mean,
                        "eligible": eligible,
                    }
                )

    eligible = [row for row in candidates if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                float(row["dev_mean"]["tie_recall"]),
                float(row["dev_mean"]["accuracy"]),
                -float(row["dev_mean"]["ece"]),
                float(row["dev_mean"]["macro_f1"]),
            ),
        )
        if eligible
        else None
    )
    policy = dict(selected["policy"]) if selected is not None else None

    per_seed: Dict[str, Any] = {}
    for seed, payload in seed_payloads.items():
        initial_dev = metrics_for_rows(payload["dev_rows"])
        initial_test = metrics_for_rows(payload["test_rows"])
        per_seed[seed] = {
            "selected_policy": policy,
            "initial_dev_metrics": initial_dev,
            "initial_test_metrics": initial_test,
            "dev_metrics": selected_or_noop_metrics(payload["dev_rows"], policy),
            "test_metrics": selected_or_noop_metrics(payload["test_rows"], policy),
            "used_noop_fallback": policy is None,
        }

    mean_std: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for key in ("initial_dev", "initial_test", "dev", "test"):
        rows = [row[f"{key}_metrics"] for row in per_seed.values()]
        mean_std[key] = summarize_rows(rows)

    checks = pass_fail(mean_std["test"], args)
    return {
        "setting": setting,
        "mode": "global_3seed_dev_mean_policy",
        "seeds": [str(seed) for seed in args.seeds],
        "selection_constraints": {
            "dataset": args.dataset,
            "dev_accuracy_min": float(args.dev_accuracy_min),
            "dev_macro_f1_min": float(args.dev_macro_f1_min),
            "dev_tie_recall_min": float(args.dev_tie_recall_min),
            "ece_max": float(args.ece_max),
            "selection_order": "maximize dev mean tie_recall, then accuracy, -ece, macro_f1",
        },
        "test_success_criteria": {
            "accuracy_min": float(args.test_accuracy_min),
            "tie_recall_strict_min": float(args.test_tie_recall_min),
            "macro_f1_min": float(args.test_macro_f1_min),
            "ece_max": float(args.test_ece_max),
        },
        "selected_policy": policy,
        "eligible_count": len(eligible),
        "candidate_count": len(candidates),
        "per_seed": per_seed,
        "mean_std": mean_std,
        "success_checks": checks,
        "passed": all(checks.values()) and policy is not None,
    }


def build_audit(args: argparse.Namespace) -> Dict[str, Any]:
    if args.selection_mode == "global":
        results = [build_global_setting_audit(str(setting), args) for setting in args.settings]
    else:
        results = [build_setting_audit(str(setting), args) for setting in args.settings]
    return {
        "status": "accuracy_constrained_tie_rescue_audit",
        "settings": [str(setting) for setting in args.settings],
        "selection_mode": args.selection_mode,
        "std_type": "sample",
        "results": results,
    }


def format_mean_std(item: Mapping[str, Optional[float]]) -> str:
    if item.get("mean") is None:
        return ""
    return f"{float(item['mean']):.4f} +/- {float(item['std'] or 0.0):.4f}"


def metrics_table(row: Mapping[str, Mapping[str, Optional[float]]]) -> str:
    return " | ".join(format_mean_std(row[metric]) for metric in METRICS)


def markdown_report(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Accuracy-Constrained Tie Rescue Audit",
        "",
        "This audit selects Tie rescue policies on dev only and applies each selected policy once to test.",
        "",
    ]
    for result in audit["results"]:
        lines.extend(
            [
                f"## {result['setting']}",
                "",
                f"Passed success criteria: `{result['passed']}`",
                f"Selection mode: `{result.get('mode', 'per_seed_policy')}`",
                f"Selected policy: `{result.get('selected_policy', 'per-seed')}`",
                "",
                "| split | accuracy | macro_f1 | ece | tie_recall |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for split in ("initial_dev", "dev", "initial_test", "test"):
            lines.append(f"| {split} | {metrics_table(result['mean_std'][split])} |")

        lines.extend(
            [
                "",
                "### Success Checks",
                "",
                "| check | passed |",
                "| --- | --- |",
            ]
        )
        for check, passed in result["success_checks"].items():
            lines.append(f"| {check} | {passed} |")

        lines.extend(
            [
                "",
                "### Per Seed",
                "",
                "| seed | threshold | margin | max_ab_conf | eligible | dev_accuracy | dev_macro_f1 | dev_ece | dev_tie_recall | test_accuracy | test_macro_f1 | test_ece | test_tie_recall | test_rescued | test_rescue_precision |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for seed, row in result["per_seed"].items():
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
                        str(policy.get("max_ab_confidence", "")),
                        str(row.get("eligible_count", "")),
                        str(dev.get("accuracy", "")),
                        str(dev.get("macro_f1", "")),
                        str(dev.get("ece", "")),
                        str(dev.get("tie_recall", "")),
                        str(test.get("accuracy", "")),
                        str(test.get("macro_f1", "")),
                        str(test.get("ece", "")),
                        str(test.get("tie_recall", "")),
                        str(test.get("rescued_count", "")),
                        str(test.get("rescued_precision", "")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit accuracy-constrained Tie rescue policies.")
    parser.add_argument("--settings", nargs="*", default=["epoch1_1024", "epoch2_1024"])
    parser.add_argument("--seeds", nargs="*", default=["13", "42", "2026"])
    parser.add_argument(
        "--calibrated-template",
        default="datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_{setting}/calibrated_results.json",
    )
    parser.add_argument(
        "--guardrail-validation-template",
        default="datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1_1024/validation_report.json",
    )
    parser.add_argument("--dataset", default="helpsteer2")
    parser.add_argument("--selection-mode", choices=["per-seed", "global"], default="per-seed")
    parser.add_argument("--ece-max", type=float, default=0.06)
    parser.add_argument("--guardrail-epsilon", type=float, default=1e-9)
    parser.add_argument("--dev-accuracy-min", type=float, default=0.8025)
    parser.add_argument("--dev-tie-recall-min", type=float, default=0.4538)
    parser.add_argument("--dev-macro-f1-min", type=float, default=0.7128)
    parser.add_argument("--test-accuracy-min", type=float, default=0.8025)
    parser.add_argument("--test-tie-recall-min", type=float, default=0.4538)
    parser.add_argument("--test-macro-f1-min", type=float, default=0.7128)
    parser.add_argument("--test-ece-max", type=float, default=0.06)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--margins", nargs="*", type=float, default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--max-ab-confidences", nargs="*", type=float, default=[0.45, 0.50, 0.55, 0.60])
    parser.add_argument(
        "--output-dir",
        default="datasets/model_outputs/accuracy_constrained_tie_rescue_3seed_summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args)
    output_dir = resolve_root_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "accuracy_constrained_tie_rescue_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "accuracy_constrained_tie_rescue_audit.md").write_text(
        markdown_report(audit),
        encoding="utf-8",
    )
    print(json.dumps({row["setting"]: row["passed"] for row in audit["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
