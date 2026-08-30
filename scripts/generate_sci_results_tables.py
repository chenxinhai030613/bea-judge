"""Generate SCI-ready BEA-Judge result tables from completed pipeline outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
MODEL_OUT = DATASETS / "model_outputs"
from path_utils import resolve_project_path
DEFAULT_BASE_SCORES = (
    DATASETS
    / "judge_outputs"
    / "m_prometheus_3b_bea10k_v2"
    / "base_scores.repaired.json"
)
DEFAULT_REPAIR_REPORT = DEFAULT_BASE_SCORES.parent / "base_scores_repair_report.json"
DEFAULT_OUTPUT_DIR = MODEL_OUT / "sci_tables"
DEFAULT_BIAS_PROFILES = DATASETS / "bias_profiles.json"
DEFAULT_EVIDENCE_PROFILES = DATASETS / "evidence_profiles.json"
DEFAULT_SWAP_REPORT = DATASETS / "judge_outputs" / "order_swap_probe" / "swap_probe_report.json"
DEFAULT_EXPANSION_REPORT = DATASETS / "expansion_v2_report.json"
DEFAULT_MANIFEST_V2 = DATASETS / "data_manifest_v2.json"

PAIRWISE_LABELS = ["A>B", "B>A", "Tie"]
FACTUALITY_LABELS = ["supported", "unsupported", "ambiguous"]
EXPECTED_ABLATIONS = {
    "Full BEA-Judge",
    "w/o Bias Module",
    "w/o Evidence Module",
    "w/o Calibration",
    "w/o Base Judge Scores",
    "w/o Tie Policy",
    "w/o Review Threshold",
}
BOOTSTRAP_SEED = 20260520
BOOTSTRAP_RESAMPLES = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_latest_calibrated_results(config_path: Path) -> Path:
    payload = load_json(config_path)
    latest = payload.get("latest_outputs", {})
    calibrated = latest.get("calibrated_results")
    if not calibrated:
        raise ValueError(f"missing latest calibrated_results in {config_path}")
    return resolve_project_path(ROOT, calibrated)


def labels_for_head(head: str) -> List[str]:
    if head == "pairwise":
        return PAIRWISE_LABELS
    if head == "factuality":
        return FACTUALITY_LABELS
    raise ValueError(f"unknown head: {head}")


def is_valid_base_score_row(row: Dict[str, Any]) -> bool:
    if row.get("judge_backend") not in {"m_prometheus", "prometheus2"}:
        return False
    if row.get("parse_status") in {"failed", "backend_error"}:
        return False
    scores = row.get("parsed_scores", {})
    return (
        row.get("id") is not None
        and row.get("pred_label") in PAIRWISE_LABELS
        and isinstance(scores.get("score_a"), (int, float))
        and isinstance(scores.get("score_b"), (int, float))
    )


def count_unresolved_rows(repair_report: Dict[str, Any]) -> int:
    value = repair_report.get("unresolved_rows", [])
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        return len(value)
    return 0


def validate_sci_gates(
    *,
    base_scores: Any,
    repair_report: Dict[str, Any],
    validation_report: Dict[str, Any],
    ablation_report: Dict[str, Any],
    bias_report: Dict[str, Any],
    evidence_report: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(base_scores, list):
        raise ValueError("base_scores.repaired.json must be a list")

    valid_ids = {str(row.get("id")) for row in base_scores if isinstance(row, dict) and is_valid_base_score_row(row)}
    coverage = repair_report.get("coverage", {})
    required = int(coverage.get("required_pairwise_rows", 0))
    covered = int(coverage.get("covered_pairwise_rows", 0))
    unresolved = count_unresolved_rows(repair_report)
    data_counts = validation_report.get("data_counts", {})
    expected_evidence_profiles = int(data_counts.get("factuality_train", 0)) + int(data_counts.get("factuality_dev", 0)) + int(data_counts.get("factuality_test", 0))
    evidence_profile_count = int(evidence_report.get("summary", {}).get("overall", {}).get("profile_count", 0))
    factuality_metrics = (
        validation_report.get("test_evaluation", {})
        .get("factuality", {})
        .get("metrics", {})
    )
    checks = {
        "base_scores_is_list": isinstance(base_scores, list),
        "base_scores_valid_pairwise_coverage_complete": required > 0 and len(valid_ids) == required,
        "repair_coverage_complete": required > 0 and covered == required,
        "repair_unresolved_zero": unresolved == 0,
        "validation_gate_passed": bool(validation_report.get("validation_gate", {}).get("passed")),
        "no_heuristic_formal_run": validation_report.get("backbone", {}).get("base_judge")
        != "heuristic_fallback_local_prototype",
        "ablation_variants_present": EXPECTED_ABLATIONS.issubset(
            {str(row.get("name")) for row in ablation_report.get("variants", [])}
        ),
        "ablation_not_local_prototype": not bool(ablation_report.get("local_prototype")),
        "bias_prediction_coverage_1": float(
            bias_report.get("prediction_coverage", {}).get("coverage_ratio", 0.0)
        )
        == 1.0,
        "bias_risk_scores_in_range": bool(
            bias_report.get("validation_gates", {}).get("risk_scores_in_range")
        ),
        "evidence_profile_count_complete": evidence_profile_count >= max(1, expected_evidence_profiles),
        "evidence_risk_scores_in_range": bool(
            evidence_report.get("validation_gates", {}).get("risk_scores_in_range")
        ),
        "factuality_macro_f1_floor_0_70": float(factuality_metrics.get("macro_f1", 0.0)) >= 0.70,
        "factuality_ece_max_0_04": float(factuality_metrics.get("ece", 1.0)) <= 0.04,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"SCI result gate failed: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "valid_pairwise_base_score_ids": len(valid_ids),
        "expected_pairwise_base_score_ids": required,
        "evidence_profile_count": evidence_profile_count,
        "expected_evidence_profiles_from_validation": expected_evidence_profiles,
    }


def review_rate(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get("review_flag")) / len(rows), 4)


def calibrated_rows(payload: Dict[str, Any], split: str, head: str) -> List[Dict[str, Any]]:
    return list(payload.get(split, {}).get(head, []))


def metric_from_report(report: Dict[str, Any], split: str, head: str) -> Dict[str, Any]:
    if split == "dev":
        return report["heads"][head]["calibrated_dev_metrics"]
    return report["test_evaluation"][head]["metrics"]


def count_from_report(report: Dict[str, Any], split: str, head: str) -> int:
    key = f"{head}_{split}"
    return int(report.get("data_counts", {}).get(key, 0))


def fmt_metric(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return round(value, 4)
    return value


def build_main_results_rows(
    validation_report: Dict[str, Any],
    calibrated: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ("dev", "test"):
        for head in ("pairwise", "factuality"):
            metrics = metric_from_report(validation_report, split, head)
            split_rows = calibrated_rows(calibrated, split, head)
            rows.append(
                {
                    "head": head,
                    "split": split,
                    "n": count_from_report(validation_report, split, head),
                    "accuracy": fmt_metric(metrics.get("accuracy")),
                    "macro_f1": fmt_metric(metrics.get("macro_f1")),
                    "ece": fmt_metric(metrics.get("ece")),
                    "brier": fmt_metric(metrics.get("brier")),
                    "tie_recall": fmt_metric(metrics.get("tie_recall")),
                    "review_rate": review_rate(split_rows),
                }
            )
    return rows


def build_ablation_rows(ablation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in ablation_report.get("variants", []):
        for head in ("pairwise", "factuality"):
            if head not in variant:
                continue
            metrics = variant[head]["test_metrics"]
            test_rows = variant[head].get("test_rows", [])
            rows.append(
                {
                    "variant": variant["name"],
                    "head": head,
                    "split": "test",
                    "n": metrics.get("n", len(test_rows)),
                    "accuracy": fmt_metric(metrics.get("accuracy")),
                    "macro_f1": fmt_metric(metrics.get("macro_f1")),
                    "ece": fmt_metric(metrics.get("ece")),
                    "brier": fmt_metric(metrics.get("brier")),
                    "tie_recall": fmt_metric(metrics.get("tie_recall")),
                    "review_rate": review_rate(test_rows),
                }
            )
    return rows


def comparison_entries(ablation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(ablation_report.get("variants", [])) + list(ablation_report.get("control_baselines", []))


def build_baseline_comparison_rows(ablation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in comparison_entries(ablation_report):
        source = "control" if entry in ablation_report.get("control_baselines", []) else "module_variant"
        for head in ("pairwise", "factuality"):
            if head not in entry:
                continue
            metrics = entry[head].get("test_metrics", {})
            test_rows = entry[head].get("test_rows", [])
            rows.append(
                {
                    "system": entry.get("name"),
                    "source": source,
                    "head": head,
                    "split": "test",
                    "n": metrics.get("n", len(test_rows)),
                    "accuracy": fmt_metric(metrics.get("accuracy")),
                    "macro_f1": fmt_metric(metrics.get("macro_f1")),
                    "ece": fmt_metric(metrics.get("ece")),
                    "brier": fmt_metric(metrics.get("brier")),
                    "tie_recall": fmt_metric(metrics.get("tie_recall")),
                    "review_rate": review_rate(test_rows),
                }
            )
    return rows


def build_evidence_feature_group_ablation_rows(ablation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in ablation_report.get("feature_group_ablations", []):
        metrics = variant.get("test_metrics", {})
        rows.append(
            {
                "feature_group": variant.get("name"),
                "weighted_calibration": variant.get("weighted_calibration"),
                "feature_count": variant.get("feature_count"),
                "accuracy": fmt_metric(metrics.get("accuracy")),
                "macro_f1": fmt_metric(metrics.get("macro_f1")),
                "ece": fmt_metric(metrics.get("ece")),
                "brier": fmt_metric(metrics.get("brier")),
            }
        )
    return rows


def build_tie_recall_rows(ablation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for variant in comparison_entries(ablation_report):
        if "pairwise" not in variant:
            continue
        for split_key, split_name in (("dev_metrics", "dev"), ("test_metrics", "test")):
            metrics = variant["pairwise"][split_key]
            gold = metrics.get("gold_distribution", {})
            pred = metrics.get("pred_distribution", {})
            rows.append(
                {
                    "variant": variant["name"],
                    "split": split_name,
                    "gold_tie": int(gold.get("Tie", 0)),
                    "pred_tie": int(pred.get("Tie", 0)),
                    "tie_recall": fmt_metric(metrics.get("tie_recall")),
                    "macro_f1": fmt_metric(metrics.get("macro_f1")),
                    "accuracy": fmt_metric(metrics.get("accuracy")),
                }
            )
    return rows


def encode_rows(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> Tuple[List[int], List[int], List[List[float]], List[float]]:
    label_to_index = {label: i for i, label in enumerate(labels)}
    y_true: List[int] = []
    y_pred: List[int] = []
    probs: List[List[float]] = []
    confidence: List[float] = []
    for row in rows:
        gold = row.get("human_label")
        pred = row.get("predicted_label")
        if gold not in label_to_index or pred not in label_to_index:
            continue
        y_true.append(label_to_index[str(gold)])
        y_pred.append(label_to_index[str(pred)])
        row_probs = row.get("label_probabilities", {})
        probs.append([float(row_probs.get(label, 0.0)) for label in labels])
        confidence.append(float(row.get("confidence", 0.0)))
    return y_true, y_pred, probs, confidence


def simple_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        return 0.0
    return round(sum(1 for gold, pred in zip(y_true, y_pred) if gold == pred) / len(y_true), 4)


def simple_macro_f1(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int]) -> float:
    scores: List[float] = []
    for label in labels:
        if label not in y_true and label not in y_pred:
            continue
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def simple_ece(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    confidence: Sequence[float],
    bins: int = 10,
) -> float:
    if not y_true:
        return 0.0
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        selected = [
            i
            for i, value in enumerate(confidence)
            if value >= lo and (value < hi if index < bins - 1 else value <= hi)
        ]
        if not selected:
            continue
        acc = sum(1 for i in selected if y_true[i] == y_pred[i]) / len(selected)
        avg_conf = sum(confidence[i] for i in selected) / len(selected)
        ece += len(selected) / total * abs(acc - avg_conf)
    return round(ece, 4)


def simple_brier(y_true: Sequence[int], probs: Sequence[Sequence[float]], label_count: int) -> float:
    if not y_true:
        return 0.0
    total = 0.0
    for gold, row_probs in zip(y_true, probs):
        for index in range(label_count):
            target = 1.0 if gold == index else 0.0
            total += (float(row_probs[index]) - target) ** 2
    return round(total / len(y_true), 4)


def simple_tie_recall(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[str]) -> Optional[float]:
    if "Tie" not in labels:
        return None
    tie_index = labels.index("Tie")
    selected = [i for i, gold in enumerate(y_true) if gold == tie_index]
    if not selected:
        return None
    return round(sum(1 for i in selected if y_pred[i] == tie_index) / len(selected), 4)


def metrics_for_rows(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    y_true, y_pred, probs, confidence = encode_rows(rows, labels)
    return {
        "n": len(y_true),
        "accuracy": simple_accuracy(y_true, y_pred),
        "macro_f1": simple_macro_f1(y_true, y_pred, list(range(len(labels)))),
        "ece": simple_ece(y_true, y_pred, confidence),
        "brier": simple_brier(y_true, probs, len(labels)),
        "tie_recall": simple_tie_recall(y_true, y_pred, labels),
        "review_rate": review_rate(rows),
        "pred_distribution": dict(Counter(row.get("predicted_label") for row in rows)),
        "gold_distribution": dict(Counter(row.get("human_label") for row in rows)),
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def bootstrap_metric_ci(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    metric_name: str,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[Any, Any]:
    if not rows:
        return "", ""
    point = metrics_for_rows(rows, labels).get(metric_name)
    if point is None or point == "":
        return "", ""
    rng = random.Random(seed + len(rows) + sum(ord(ch) for ch in metric_name))
    values: List[float] = []
    row_list = list(rows)
    for _ in range(resamples):
        sample = [row_list[rng.randrange(len(row_list))] for _ in row_list]
        value = metrics_for_rows(sample, labels).get(metric_name)
        if value is not None and value != "":
            values.append(float(value))
    return round(percentile(values, 0.025), 4), round(percentile(values, 0.975), 4)


def build_metric_ci_rows(calibrated: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ("dev", "test"):
        for head in ("pairwise", "factuality"):
            split_rows = calibrated_rows(calibrated, split, head)
            labels = labels_for_head(head)
            metrics = metrics_for_rows(split_rows, labels)
            for metric_name in ("accuracy", "macro_f1", "ece", "brier", "tie_recall"):
                value = metrics.get(metric_name)
                ci_low, ci_high = bootstrap_metric_ci(split_rows, labels, metric_name)
                rows.append(
                    {
                        "head": head,
                        "split": split,
                        "metric": metric_name,
                        "point": fmt_metric(value),
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "n": metrics.get("n", 0),
                    }
                )
    return rows


def compact_prediction_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id"))
        if sample_id:
            out[sample_id] = row
    return out


def mcnemar_exact_or_chi2_pvalue(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    if n <= 200:
        tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
        return round(min(1.0, 2.0 * tail), 6)
    statistic = ((abs(b - c) - 1.0) ** 2) / n
    return round(math.erfc(math.sqrt(statistic / 2.0)), 6)


def paired_bootstrap_delta_ci(
    aligned: Sequence[Tuple[int, int, int]],
    labels: Sequence[str],
    metric_name: str,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    if not aligned:
        return 0.0, 0.0
    rng = random.Random(seed + len(aligned) + sum(ord(ch) for ch in metric_name))
    labels_idx = list(range(len(labels)))
    deltas: List[float] = []
    for _ in range(resamples):
        sampled = [aligned[rng.randrange(len(aligned))] for _ in aligned]
        y_true = [row[0] for row in sampled]
        full_pred = [row[1] for row in sampled]
        variant_pred = [row[2] for row in sampled]
        if metric_name == "accuracy":
            full_value = simple_accuracy(y_true, full_pred)
            variant_value = simple_accuracy(y_true, variant_pred)
        else:
            full_value = simple_macro_f1(y_true, full_pred, labels_idx)
            variant_value = simple_macro_f1(y_true, variant_pred, labels_idx)
        deltas.append(float(full_value) - float(variant_value))
    return round(percentile(deltas, 0.025), 4), round(percentile(deltas, 0.975), 4)


def build_ablation_significance_rows(ablation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = {str(row.get("name")): row for row in comparison_entries(ablation_report)}
    full = variants.get("Full BEA-Judge")
    if not full:
        return []
    out: List[Dict[str, Any]] = []
    for variant_name, variant in sorted(variants.items()):
        if variant_name == "Full BEA-Judge":
            continue
        for head in ("pairwise", "factuality"):
            if head not in variant or head not in full:
                continue
            labels = labels_for_head(head)
            label_to_index = {label: i for i, label in enumerate(labels)}
            full_rows = compact_prediction_rows(full.get(head, {}).get("test_rows", []))
            variant_rows = compact_prediction_rows(variant.get(head, {}).get("test_rows", []))
            aligned: List[Tuple[int, int, int]] = []
            b = 0
            c = 0
            for sample_id in sorted(set(full_rows) & set(variant_rows)):
                full_row = full_rows[sample_id]
                variant_row = variant_rows[sample_id]
                gold = full_row.get("human_label")
                full_pred = full_row.get("predicted_label")
                variant_pred = variant_row.get("predicted_label")
                if gold not in label_to_index or full_pred not in label_to_index or variant_pred not in label_to_index:
                    continue
                full_correct = full_pred == gold
                variant_correct = variant_pred == gold
                if full_correct and not variant_correct:
                    b += 1
                elif variant_correct and not full_correct:
                    c += 1
                aligned.append(
                    (
                        label_to_index[str(gold)],
                        label_to_index[str(full_pred)],
                        label_to_index[str(variant_pred)],
                    )
                )
            full_metrics = full.get(head, {}).get("test_metrics", {})
            variant_metrics = variant.get(head, {}).get("test_metrics", {})
            acc_low, acc_high = paired_bootstrap_delta_ci(aligned, labels, "accuracy")
            f1_low, f1_high = paired_bootstrap_delta_ci(aligned, labels, "macro_f1")
            out.append(
                {
                    "variant": variant_name,
                    "head": head,
                    "paired_n": len(aligned),
                    "full_accuracy": fmt_metric(full_metrics.get("accuracy")),
                    "variant_accuracy": fmt_metric(variant_metrics.get("accuracy")),
                    "delta_accuracy_full_minus_variant": round(
                        float(full_metrics.get("accuracy", 0.0)) - float(variant_metrics.get("accuracy", 0.0)),
                        4,
                    ),
                    "delta_accuracy_ci95_low": acc_low,
                    "delta_accuracy_ci95_high": acc_high,
                    "full_macro_f1": fmt_metric(full_metrics.get("macro_f1")),
                    "variant_macro_f1": fmt_metric(variant_metrics.get("macro_f1")),
                    "delta_macro_f1_full_minus_variant": round(
                        float(full_metrics.get("macro_f1", 0.0)) - float(variant_metrics.get("macro_f1", 0.0)),
                        4,
                    ),
                    "delta_macro_f1_ci95_low": f1_low,
                    "delta_macro_f1_ci95_high": f1_high,
                    "mcnemar_full_only_correct": b,
                    "mcnemar_variant_only_correct": c,
                    "mcnemar_p": mcnemar_exact_or_chi2_pvalue(b, c),
                }
            )
    return out


def build_bias_risk_utility_rows(ablation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in ablation_report.get("bias_utility", []):
        rows.append(
            {
                "setting": row.get("setting"),
                "head": row.get("head", "pairwise"),
                "split": row.get("split", "test"),
                "n": row.get("n", 0),
                "accuracy": fmt_metric(row.get("accuracy")),
                "macro_f1": fmt_metric(row.get("macro_f1")),
                "ece": fmt_metric(row.get("ece")),
                "review_rate": fmt_metric(row.get("review_rate")),
                "review_capture_rate": fmt_metric(row.get("review_capture_rate")),
            }
        )
    return rows


def build_risk_coverage_rows(calibrated: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for head in ("pairwise", "factuality"):
        split_rows = calibrated_rows(calibrated, "test", head)
        if not split_rows:
            continue
        ordered = sorted(split_rows, key=lambda row: float(row.get("risk_score", 0.0)), reverse=True)
        total_errors = sum(1 for row in ordered if row.get("human_label") != row.get("predicted_label"))
        for coverage in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00):
            review_count = min(len(ordered), max(1, int(round(len(ordered) * coverage))))
            reviewed = ordered[:review_count]
            auto = ordered[review_count:]
            captured_errors = sum(1 for row in reviewed if row.get("human_label") != row.get("predicted_label"))
            residual_errors = sum(1 for row in auto if row.get("human_label") != row.get("predicted_label"))
            rows.append(
                {
                    "head": head,
                    "split": "test",
                    "review_rate": round(review_count / len(ordered), 4),
                    "review_count": review_count,
                    "error_capture_rate": round(captured_errors / total_errors, 4) if total_errors else "",
                    "auto_accept_count": len(auto),
                    "auto_accept_accuracy": round(1.0 - residual_errors / len(auto), 4) if auto else "",
                    "risk_threshold": round(float(reviewed[-1].get("risk_score", 0.0)), 6) if reviewed else "",
                }
            )
    return rows


def build_calibration_method_rows(calibration_summary: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not calibration_summary:
        return []
    rows: List[Dict[str, Any]] = []
    for method, payload in sorted(calibration_summary.get("results", {}).items()):
        for split, metrics_key in (("dev", "metrics_dev"), ("test", "metrics_test")):
            metrics = payload.get(metrics_key, {})
            extras = payload.get("extras", {})
            rows.append(
                {
                    "method": method,
                    "split": split,
                    "accuracy": fmt_metric(metrics.get("accuracy")),
                    "ece": fmt_metric(metrics.get("ece")),
                    "mce": fmt_metric(metrics.get("mce")),
                    "brier": fmt_metric(metrics.get("brier")),
                    "nll": fmt_metric(metrics.get("nll")),
                    "coverage": fmt_metric(extras.get(f"coverage_{split}")),
                    "set_size_avg": fmt_metric(extras.get("set_size_avg_test")) if split == "test" else "",
                }
            )
    return rows


def build_per_dataset_rows(calibrated: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for head in ("pairwise", "factuality"):
        by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in calibrated_rows(calibrated, "test", head):
            by_dataset[str(row.get("dataset", "unknown"))].append(row)
        for dataset in sorted(by_dataset):
            metrics = metrics_for_rows(by_dataset[dataset], labels_for_head(head))
            rows.append(
                {
                    "head": head,
                    "dataset": dataset,
                    "n": metrics["n"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "ece": metrics["ece"],
                    "brier": metrics["brier"],
                    "tie_recall": fmt_metric(metrics["tie_recall"]),
                    "review_rate": metrics["review_rate"],
                }
            )
    return rows


def build_ragtruth_result_rows(calibrated: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ("dev", "test"):
        split_rows = [
            row
            for row in calibrated_rows(calibrated, split, "factuality")
            if str(row.get("dataset")) == "ragtruth"
        ]
        metrics = metrics_for_rows(split_rows, FACTUALITY_LABELS)
        confusion_counts = Counter(
            f"{row.get('human_label')}->{row.get('predicted_label')}"
            for row in split_rows
        )
        rows.append(
            {
                "split": split,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "ece": metrics["ece"],
                "brier": metrics["brier"],
                "review_rate": metrics["review_rate"],
                "supported_to_unsupported": confusion_counts.get("supported->unsupported", 0),
                "unsupported_to_supported": confusion_counts.get("unsupported->supported", 0),
                "ambiguous_errors": sum(
                    count
                    for key, count in confusion_counts.items()
                    if key.startswith("ambiguous->") and key != "ambiguous->ambiguous"
                ),
            }
        )
    return rows


def base_score_map(base_scores: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("id")): row for row in base_scores if isinstance(row, dict) and is_valid_base_score_row(row)}


def build_base_diagnostic_rows(
    base_scores: Sequence[Dict[str, Any]],
    calibrated: Dict[str, Any],
) -> List[Dict[str, Any]]:
    base_by_id = base_score_map(base_scores)
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for split in ("dev", "test"):
        for row in calibrated_rows(calibrated, split, "pairwise"):
            key = (
                split,
                str(row.get("dataset", "unknown")),
                str(row.get("task_type", "unknown")),
                str(row.get("human_label", "unknown")),
            )
            groups[key].append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups):
        split, dataset, task_type, gold_label = key
        rows = groups[key]
        n = len(rows)
        base_correct = 0
        calibrated_correct = 0
        disagreements = 0
        base_tie = 0
        calibrated_tie = 0
        margins: List[float] = []
        conflicts = 0
        for row in rows:
            base = base_by_id.get(str(row.get("id")), {})
            base_label = base.get("pred_label")
            pred_label = row.get("predicted_label")
            if base_label == row.get("human_label"):
                base_correct += 1
            if pred_label == row.get("human_label"):
                calibrated_correct += 1
            if base_label != pred_label:
                disagreements += 1
            if base_label == "Tie":
                base_tie += 1
            if pred_label == "Tie":
                calibrated_tie += 1
            if base_label and base_label != row.get("human_label"):
                conflicts += 1
            scores = base.get("parsed_scores", {})
            if isinstance(scores.get("score_a"), (int, float)) and isinstance(scores.get("score_b"), (int, float)):
                margins.append(abs(float(scores["score_a"]) - float(scores["score_b"])))
        out.append(
            {
                "split": split,
                "dataset": dataset,
                "task_type": task_type,
                "gold_label": gold_label,
                "n": n,
                "base_accuracy": round(base_correct / n, 4) if n else 0.0,
                "calibrated_accuracy": round(calibrated_correct / n, 4) if n else 0.0,
                "base_calibrated_disagreement_rate": round(disagreements / n, 4) if n else 0.0,
                "base_tie_rate": round(base_tie / n, 4) if n else 0.0,
                "calibrated_tie_rate": round(calibrated_tie / n, 4) if n else 0.0,
                "avg_base_margin": round(sum(margins) / len(margins), 4) if margins else "",
                "base_conflict_rate": round(conflicts / n, 4) if n else 0.0,
            }
        )
    return out


def prediction_rows_from_bias_profiles(bias_profiles: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for profile in bias_profiles.get("profiles", []):
        prediction = profile.get("prediction") or {}
        pred = prediction.get("predicted_label")
        gold = prediction.get("gold_label")
        if pred not in PAIRWISE_LABELS or gold not in PAIRWISE_LABELS:
            continue
        rows[str(profile.get("id"))] = {
            "id": profile.get("id"),
            "dataset": profile.get("dataset"),
            "task_type": profile.get("task_type"),
            "split": profile.get("split"),
            "human_label": gold,
            "predicted_label": pred,
            "confidence": float(prediction.get("confidence", 0.0)),
            "review_flag": bool(profile.get("bias", {}).get("review_required")),
            "label_probabilities": {label: 1.0 if label == pred else 0.0 for label in PAIRWISE_LABELS},
        }
    return rows


def bias_group(profile: Dict[str, Any]) -> str:
    metadata = profile.get("metadata", {})
    candidates = [
        str(metadata.get("bias_type") or ""),
        str(metadata.get("perturbation_applied") or ""),
        str(metadata.get("reasoning_difficulty") or ""),
        str(metadata.get("difficulty_type") or ""),
    ]
    for value in candidates:
        normalized = value.strip()
        if normalized in {"position", "length", "format", "rubric_sensitivity", "reasoning_difficulty"}:
            return normalized
        if normalized in {"reasoning", "hard_reasoning", "complex_reasoning"}:
            return "reasoning_difficulty"
    return "none"


def build_bias_subgroup_rows(bias_profiles: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows_by_id = prediction_rows_from_bias_profiles(bias_profiles)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    risks: Dict[str, List[float]] = defaultdict(list)
    required_groups = {"position", "length", "format", "rubric_sensitivity", "reasoning_difficulty", "none"}
    for profile in bias_profiles.get("profiles", []):
        group = bias_group(profile)
        prediction = rows_by_id.get(str(profile.get("id")))
        if prediction is not None:
            groups[group].append(prediction)
        risks[group].append(float(profile.get("bias", {}).get("overall_bias_risk", 0.0)))

    out: List[Dict[str, Any]] = []
    for group in sorted(set(groups) | set(risks) | required_groups):
        metrics = metrics_for_rows(groups.get(group, []), PAIRWISE_LABELS)
        risk_values = risks.get(group, [])
        out.append(
            {
                "bias_group": group,
                "n": metrics["n"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "ece": metrics["ece"],
                "review_rate": metrics["review_rate"],
                "avg_bias_risk": round(sum(risk_values) / len(risk_values), 4) if risk_values else "",
            }
        )
    return out


def prediction_rows_from_calibrated(calibrated: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        for head in ("pairwise", "factuality"):
            for row in calibrated_rows(calibrated, split, head):
                rows[str(row.get("id"))] = row
    return rows


def build_evidence_subtype_rows(
    evidence_profiles: Dict[str, Any],
    calibrated: Dict[str, Any],
) -> List[Dict[str, Any]]:
    predictions = prediction_rows_from_calibrated(calibrated)
    subtype_patterns = {
        "low_context_support": "low_context_support",
        "reference_support_gap": "reference_support_gap",
        "numeric_evidence_gap": "numeric_evidence_gap",
        "date_evidence_gap": "date_evidence_gap",
        "entity_gap": "entity_evidence_gap",
        "entity_alias_gap": "entity_alias_gap",
        "low_support_sentence_ratio": "low_support_sentence_ratio",
        "low_support_anchor_sentence_ratio": "low_support_anchor_sentence_ratio",
        "max_low_support_anchor_gap": "max_low_support_anchor_gap",
        "anchored_hallucination_severity": "anchored_hallucination_severity",
        "local_hallucination_risk": "local_hallucination_risk",
        "negation_mismatch": "negation_mismatch",
        "comparative_mismatch": "comparative_mismatch",
    }
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    risks: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for profile in evidence_profiles.get("profiles", []):
        reasons = profile.get("evidence", {}).get("reasons", [])
        matched = [name for name, pattern in subtype_patterns.items() if any(pattern in str(reason) for reason in reasons)]
        if not matched:
            matched = ["none"]
        prediction = predictions.get(str(profile.get("id")))
        for subtype in matched:
            if prediction is not None:
                head = str(prediction.get("head"))
                key = (subtype, head)
                groups[key].append(prediction)
                risks[key].append(float(profile.get("evidence", {}).get("evidence_risk", 0.0)))

    out: List[Dict[str, Any]] = []
    required = {(subtype, "factuality") for subtype in subtype_patterns}
    for subtype, head in sorted(set(groups) | set(risks) | required):
        rows = groups.get((subtype, head), [])
        labels = labels_for_head(head)
        metrics = metrics_for_rows(rows, labels) if rows else {"n": 0, "accuracy": 0.0, "review_rate": 0.0}
        errors = [row for row in rows if row.get("human_label") != row.get("predicted_label")]
        reviewed_errors = [row for row in errors if row.get("review_flag")]
        risk_values = risks.get((subtype, head), [])
        out.append(
            {
                "evidence_subtype": subtype,
                "head": head,
                "n": metrics["n"],
                "error_rate": round(1.0 - float(metrics["accuracy"]), 4) if metrics["n"] else 0.0,
                "review_capture_rate": round(len(reviewed_errors) / len(errors), 4) if errors else "",
                "review_rate": metrics["review_rate"],
                "avg_evidence_risk": round(sum(risk_values) / len(risk_values), 4) if risk_values else "",
                "error_count": len(errors),
            }
        )
    return out


def build_swap_consistency_rows(swap_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in swap_report.get("dataset_summary", []):
        rows.append(
            {
                "dataset": row.get("dataset"),
                "selected_n": row.get("selected_n", 0),
                "swap_available_n": row.get("swap_available_n", 0),
                "swap_parse_success_rate": row.get("swap_parse_success_rate", 0.0),
                "swap_consistency_rate": row.get("swap_consistency_rate", ""),
                "swap_inconsistency_rate": row.get("swap_inconsistency_rate", ""),
                "calibrated_error_rate": row.get("calibrated_error_rate", 0.0),
                "error_rate_when_inconsistent": row.get("error_rate_when_inconsistent", ""),
                "avg_swap_margin_delta": row.get("avg_swap_margin_delta", ""),
                "tie_case_rate": row.get("tie_case_rate", 0.0),
            }
        )
    return rows


def build_source_provenance_rows(manifest: Dict[str, Any], expansion_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    accepted_by_source = {
        str(row.get("source")): int(row.get("accepted_records", 0))
        for row in expansion_report.get("source_reports", [])
    }
    rows: List[Dict[str, Any]] = []
    for key, item in sorted(manifest.get("sources", {}).items()):
        rows.append(
            {
                "source": key,
                "license": item.get("license", ""),
                "redistribution_allowed": item.get("redistribution_allowed", ""),
                "admission_allowed": item.get("admission_allowed", ""),
                "admission_reason": item.get("admission_reason", ""),
                "accepted_records": accepted_by_source.get(str(key), 0),
                "acquisition_date": item.get("acquisition_date") or item.get("acquired_at") or "",
                "sha256_present": bool(item.get("sha256")),
            }
        )
    return rows


def build_license_audit_rows(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, item in sorted(manifest.get("sources", {}).items()):
        acquisition_complete = bool(item.get("acquisition_date") or item.get("acquired_at"))
        sha256_present = bool(item.get("sha256"))
        admission_allowed = bool(item.get("admission_allowed"))
        external_eval_only = bool(item.get("external_eval_only"))
        risk_flags: List[str] = []
        if not item.get("license"):
            risk_flags.append("missing_license")
        if not bool(item.get("redistribution_allowed")):
            risk_flags.append("redistribution_restricted")
        if external_eval_only:
            risk_flags.append("external_eval_only")
        if admission_allowed and not acquisition_complete:
            risk_flags.append("missing_acquisition_date")
        if admission_allowed and not sha256_present:
            risk_flags.append("missing_sha256")
        rows.append(
            {
                "source": key,
                "license": item.get("license", ""),
                "license_status": item.get("license_status", ""),
                "redistribution_allowed": item.get("redistribution_allowed", ""),
                "external_eval_only": external_eval_only,
                "admission_allowed": admission_allowed,
                "admission_reason": item.get("admission_reason", ""),
                "acquisition_complete": acquisition_complete,
                "sha256_present": sha256_present,
                "risk_flags": ";".join(risk_flags) if risk_flags else "none",
            }
        )
    return rows


def build_expansion_distribution_rows(expansion_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats = expansion_report.get("statistics", {})
    rows: List[Dict[str, Any]] = []
    for field in ("by_task_type", "by_dataset", "by_split", "by_language", "human_label_distribution"):
        for name, count in sorted((stats.get(field) or {}).items()):
            rows.append({"dimension": field, "value": name, "count": count})
    return rows


def build_data_scaling_rows(validation_report: Dict[str, Any], expansion_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    stats = expansion_report.get("statistics", {})
    pair = validation_report.get("test_evaluation", {}).get("pairwise", {}).get("metrics", {})
    fact = validation_report.get("test_evaluation", {}).get("factuality", {}).get("metrics", {})
    total_samples = int(stats.get("total_samples", 0) or 0)
    current_name = "BEA-Judge-10K-v2/formal" if total_samples >= 9500 else "BEA-Judge-current/formal"
    return [
        {
            "dataset_version": current_name,
            "sample_count": total_samples,
            "pairwise_macro_f1": fmt_metric(pair.get("macro_f1")),
            "pairwise_tie_recall": fmt_metric(pair.get("tie_recall")),
            "factuality_accuracy": fmt_metric(fact.get("accuracy")),
            "factuality_macro_f1": fmt_metric(fact.get("macro_f1")),
            "factuality_ece": fmt_metric(fact.get("ece")),
        },
        {
            "dataset_version": "BEA-Judge-3.4K/legacy-anchor",
            "sample_count": 3400,
            "pairwise_macro_f1": "",
            "pairwise_tie_recall": "",
            "factuality_accuracy": "",
            "factuality_macro_f1": "",
            "factuality_ece": "",
        },
    ]


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_table(output_dir: Path, stem: str, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> Dict[str, str]:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    write_csv(csv_path, rows, columns)
    md_path.write_text(markdown_table(rows, columns), encoding="utf-8")
    return {"csv": path_relative_to_root(csv_path), "md": path_relative_to_root(md_path)}


def build_method_summary(
    *,
    repair_report: Dict[str, Any],
    validation_report: Dict[str, Any],
    ablation_rows: Sequence[Dict[str, Any]],
    bias_report: Dict[str, Any],
    evidence_report: Dict[str, Any],
) -> str:
    pair = validation_report["test_evaluation"]["pairwise"]["metrics"]
    fact = validation_report["test_evaluation"]["factuality"]["metrics"]
    coverage = repair_report["coverage"]
    no_evidence = next(
        row for row in ablation_rows if row["variant"] == "w/o Evidence Module" and row["head"] == "factuality"
    )
    no_bias = next(row for row in ablation_rows if row["variant"] == "w/o Bias Module" and row["head"] == "pairwise")
    return "\n".join(
        [
            "# BEA-Judge SCI Method and Result Summary",
            "",
            f"Created at: {utc_now()}",
            "",
            "## Formal Model Positioning",
            "",
            "BEA-Judge is a bias-aware and evidence-augmented judge calibration framework built on top of real M-Prometheus-3B outputs. It is not a newly fine-tuned large language model.",
            "",
            "## Four-Module Pipeline",
            "",
            "1. Base Judge: M-Prometheus-3B scores all pairwise samples and emits `base_scores.repaired.json`.",
            "2. Bias Awareness: position, length, format, and rubric-sensitivity risk features are used for calibration and review prioritization.",
            "3. Evidence Enhancement: deterministic context/reference overlap, sentence support, numeric/date/entity gaps, negation/comparative mismatch, pairwise support delta, and evidence-risk features support factuality scoring.",
            "4. Fusion Calibration: task-specific softmax heads use dev-only temperature scaling, review thresholds, and pairwise Tie policy.",
            "",
            "## Reproducibility Gates",
            "",
            f"- Pairwise base-score coverage: {coverage['covered_pairwise_rows']} / {coverage['required_pairwise_rows']}.",
            f"- Repaired parse failures: {repair_report['replaced_rows']} replaced; unresolved rows: {count_unresolved_rows(repair_report)}.",
            f"- Bias prediction coverage: {bias_report['prediction_coverage']['coverage_ratio']}.",
            f"- Evidence profile count: {evidence_report['summary']['overall']['profile_count']}.",
            f"- Validation gate passed: {validation_report['validation_gate']['passed']}.",
            "",
            "## Main Results",
            "",
            f"- Pairwise test: accuracy={pair['accuracy']}, macro-F1={pair['macro_f1']}, ECE={pair['ece']}, Brier={pair['brier']}, Tie recall={pair.get('tie_recall')}.",
            f"- Factuality test: accuracy={fact['accuracy']}, macro-F1={fact['macro_f1']}, ECE={fact['ece']}, Brier={fact['brier']}.",
            f"- Without Evidence: factuality accuracy={no_evidence['accuracy']}, ECE={no_evidence['ece']}.",
            f"- Without Bias: pairwise accuracy={no_bias['accuracy']}; this supports positioning the bias module as risk identification and review prioritization.",
            "",
            "## Reporting Constraints",
            "",
            "The formal SCI result should not claim LLM fine-tuning, external retrieval, atomic claim verification, or heuristic fallback use. Bias should be reported as a risk-control module; evidence enhancement is the main factuality reliability contribution.",
            "",
        ]
    )


def build_data_availability_statement(
    *,
    manifest_v2: Optional[Dict[str, Any]],
    expansion_report: Optional[Dict[str, Any]],
    validation_report: Dict[str, Any],
    base_scores_path: Path,
    calibrated_results_path: Path,
) -> str:
    stats = (expansion_report or {}).get("statistics", {})
    data_counts = validation_report.get("data_counts", {})
    admitted_sources: List[str] = []
    external_sources: List[str] = []
    if manifest_v2:
        for name, item in sorted(manifest_v2.get("sources", {}).items()):
            if item.get("admission_allowed"):
                admitted_sources.append(f"{name} ({item.get('license', 'license not recorded')})")
            if item.get("external_eval_only"):
                external_sources.append(f"{name} ({item.get('license', 'mixed or restricted license')})")

    return "\n".join(
        [
            "# Data Availability Draft",
            "",
            "The BEA-Judge-10K v2 processed dataset, split files, model outputs, validation reports, and SCI-ready tables are generated within this repository. The current processed dataset contains "
            f"{stats.get('total_samples', sum(data_counts.get(key, 0) for key in ('train', 'dev', 'test')))} records, with train/dev/test counts "
            f"{data_counts.get('train')} / {data_counts.get('dev')} / {data_counts.get('test')}.",
            "",
            "Public source data were admitted only after license and provenance checks. Training-admitted sources recorded in the manifest are: "
            + ("; ".join(admitted_sources) if admitted_sources else "not listed in the provided manifest.")
            + " Each admitted source should retain acquisition date, revision or version reference, source URL, license, and SHA-256 metadata in `datasets/data_manifest_v2.json`.",
            "",
            "RewardBench and any mixed-license or redistribution-restricted sources are treated as external-evaluation-only resources and are not mixed into the train/dev/test splits unless their subset-level licenses are separately verified.",
            "",
            f"Formal base judge scores are stored at `{path_relative_to_root(base_scores_path)}` and use real Prometheus-family outputs only. The calibrated model predictions are stored at `{path_relative_to_root(calibrated_results_path)}`. Heuristic fallback outputs are excluded from formal SCI results.",
            "",
            "No external retrieval index, private human subject data, or closed LLM-generated labels are required to reproduce the reported v2 calibration results. If the processed data package is redistributed, the source-level license terms and attribution requirements must be preserved.",
            "",
        ]
    )


def generate_outputs(
    *,
    base_scores_path: Path,
    repair_report_path: Path,
    validation_report_path: Path,
    calibrated_results_path: Path,
    ablation_report_path: Path,
    bias_report_path: Path,
    bias_profiles_path: Path,
    evidence_report_path: Path,
    evidence_profiles_path: Path,
    output_dir: Path,
    swap_report_path: Optional[Path] = None,
    expansion_report_path: Optional[Path] = None,
    manifest_v2_path: Optional[Path] = None,
    calibration_comparison_path: Optional[Path] = None,
) -> Dict[str, Any]:
    base_scores = load_json(base_scores_path)
    repair_report = load_json(repair_report_path)
    validation_report = load_json(validation_report_path)
    calibrated = load_json(calibrated_results_path)
    ablation_report = load_json(ablation_report_path)
    bias_report = load_json(bias_report_path)
    bias_profiles = load_json(bias_profiles_path)
    evidence_report = load_json(evidence_report_path)
    evidence_profiles = load_json(evidence_profiles_path)
    swap_report = load_json(swap_report_path) if swap_report_path and swap_report_path.exists() else None
    expansion_report = load_json(expansion_report_path) if expansion_report_path and expansion_report_path.exists() else None
    manifest_v2 = load_json(manifest_v2_path) if manifest_v2_path and manifest_v2_path.exists() else None
    calibration_summary = (
        load_json(calibration_comparison_path)
        if calibration_comparison_path and calibration_comparison_path.exists()
        else None
    )
    if calibration_summary is None:
        calibration_summary = ablation_report.get("calibration_methods") or None
    gates = validate_sci_gates(
        base_scores=base_scores,
        repair_report=repair_report,
        validation_report=validation_report,
        ablation_report=ablation_report,
        bias_report=bias_report,
        evidence_report=evidence_report,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = build_main_results_rows(validation_report, calibrated)
    baseline_comparison_rows = build_baseline_comparison_rows(ablation_report)
    metric_ci_rows = build_metric_ci_rows(calibrated)
    ablation_rows = build_ablation_rows(ablation_report)
    ablation_significance_rows = build_ablation_significance_rows(ablation_report)
    bias_risk_utility_rows = build_bias_risk_utility_rows(ablation_report)
    risk_coverage_rows = build_risk_coverage_rows(calibrated)
    calibration_method_rows = build_calibration_method_rows(calibration_summary)
    evidence_feature_group_rows = build_evidence_feature_group_ablation_rows(ablation_report)
    tie_rows = build_tie_recall_rows(ablation_report)
    per_dataset_rows = build_per_dataset_rows(calibrated)
    ragtruth_rows = build_ragtruth_result_rows(calibrated)
    base_diagnostic_rows = build_base_diagnostic_rows(base_scores, calibrated)
    bias_subgroup_rows = build_bias_subgroup_rows(bias_profiles)
    evidence_subtype_rows = build_evidence_subtype_rows(evidence_profiles, calibrated)
    swap_consistency_rows = build_swap_consistency_rows(swap_report) if swap_report else []

    tables = {
        "main_results": write_table(
            output_dir,
            "main_results_table",
            main_rows,
            ["head", "split", "n", "accuracy", "macro_f1", "ece", "brier", "tie_recall", "review_rate"],
        ),
        "baseline_comparison": write_table(
            output_dir,
            "baseline_comparison_table",
            baseline_comparison_rows,
            ["system", "source", "head", "split", "n", "accuracy", "macro_f1", "ece", "brier", "tie_recall", "review_rate"],
        ),
        "metric_confidence_intervals": write_table(
            output_dir,
            "metric_confidence_interval_table",
            metric_ci_rows,
            ["head", "split", "metric", "point", "ci95_low", "ci95_high", "n"],
        ),
        "ablation": write_table(
            output_dir,
            "ablation_table",
            ablation_rows,
            ["variant", "head", "split", "n", "accuracy", "macro_f1", "ece", "brier", "tie_recall", "review_rate"],
        ),
        "ablation_significance": write_table(
            output_dir,
            "ablation_significance_table",
            ablation_significance_rows,
            [
                "variant",
                "head",
                "paired_n",
                "delta_accuracy_full_minus_variant",
                "delta_accuracy_ci95_low",
                "delta_accuracy_ci95_high",
                "delta_macro_f1_full_minus_variant",
                "delta_macro_f1_ci95_low",
                "delta_macro_f1_ci95_high",
                "mcnemar_full_only_correct",
                "mcnemar_variant_only_correct",
                "mcnemar_p",
            ],
        ),
        "evidence_feature_group_ablation": write_table(
            output_dir,
            "evidence_feature_group_ablation_table",
            evidence_feature_group_rows,
            ["feature_group", "weighted_calibration", "feature_count", "accuracy", "macro_f1", "ece", "brier"],
        ),
        "bias_risk_utility": write_table(
            output_dir,
            "bias_risk_utility_table",
            bias_risk_utility_rows,
            ["setting", "head", "split", "n", "accuracy", "macro_f1", "ece", "review_rate", "review_capture_rate"],
        ),
        "risk_coverage": write_table(
            output_dir,
            "risk_coverage_table",
            risk_coverage_rows,
            ["head", "split", "review_rate", "review_count", "error_capture_rate", "auto_accept_count", "auto_accept_accuracy", "risk_threshold"],
        ),
        "calibration_methods": write_table(
            output_dir,
            "calibration_methods_table",
            calibration_method_rows,
            ["method", "split", "accuracy", "ece", "mce", "brier", "nll", "coverage", "set_size_avg"],
        ),
        "tie_recall": write_table(
            output_dir,
            "tie_recall_table",
            tie_rows,
            ["variant", "split", "gold_tie", "pred_tie", "tie_recall", "macro_f1", "accuracy"],
        ),
        "per_dataset": write_table(
            output_dir,
            "per_dataset_table",
            per_dataset_rows,
            ["head", "dataset", "n", "accuracy", "macro_f1", "ece", "brier", "tie_recall", "review_rate"],
        ),
        "ragtruth_results": write_table(
            output_dir,
            "ragtruth_results_table",
            ragtruth_rows,
            [
                "split",
                "n",
                "accuracy",
                "macro_f1",
                "ece",
                "brier",
                "review_rate",
                "supported_to_unsupported",
                "unsupported_to_supported",
                "ambiguous_errors",
            ],
        ),
        "base_diagnostics": write_table(
            output_dir,
            "base_diagnostics_table",
            base_diagnostic_rows,
            [
                "split",
                "dataset",
                "task_type",
                "gold_label",
                "n",
                "base_accuracy",
                "calibrated_accuracy",
                "base_calibrated_disagreement_rate",
                "base_tie_rate",
                "calibrated_tie_rate",
                "avg_base_margin",
                "base_conflict_rate",
            ],
        ),
        "bias_subgroups": write_table(
            output_dir,
            "bias_subgroup_table",
            bias_subgroup_rows,
            ["bias_group", "n", "accuracy", "macro_f1", "ece", "review_rate", "avg_bias_risk"],
        ),
        "evidence_subtypes": write_table(
            output_dir,
            "evidence_subtype_table",
            evidence_subtype_rows,
            [
                "evidence_subtype",
                "head",
                "n",
                "error_rate",
                "review_capture_rate",
                "review_rate",
                "avg_evidence_risk",
                "error_count",
            ],
        ),
    }
    if swap_report:
        tables["swap_consistency"] = write_table(
            output_dir,
            "swap_consistency_table",
            swap_consistency_rows,
            [
                "dataset",
                "selected_n",
                "swap_available_n",
                "swap_parse_success_rate",
                "swap_consistency_rate",
                "swap_inconsistency_rate",
                "calibrated_error_rate",
                "error_rate_when_inconsistent",
                "avg_swap_margin_delta",
                "tie_case_rate",
            ],
        )
    if expansion_report and manifest_v2:
        tables["source_provenance"] = write_table(
            output_dir,
            "source_provenance_table",
            build_source_provenance_rows(manifest_v2, expansion_report),
            [
                "source",
                "license",
                "redistribution_allowed",
                "admission_allowed",
                "admission_reason",
                "accepted_records",
                "acquisition_date",
                "sha256_present",
            ],
        )
        tables["license_audit"] = write_table(
            output_dir,
            "license_audit_table",
            build_license_audit_rows(manifest_v2),
            [
                "source",
                "license",
                "license_status",
                "redistribution_allowed",
                "external_eval_only",
                "admission_allowed",
                "admission_reason",
                "acquisition_complete",
                "sha256_present",
                "risk_flags",
            ],
        )
        tables["v2_distribution"] = write_table(
            output_dir,
            "v2_distribution_table",
            build_expansion_distribution_rows(expansion_report),
            ["dimension", "value", "count"],
        )
        tables["data_scaling"] = write_table(
            output_dir,
            "data_scaling_table",
            build_data_scaling_rows(validation_report, expansion_report),
            [
                "dataset_version",
                "sample_count",
                "pairwise_macro_f1",
                "pairwise_tie_recall",
                "factuality_accuracy",
                "factuality_macro_f1",
                "factuality_ece",
            ],
        )

    method_summary = build_method_summary(
        repair_report=repair_report,
        validation_report=validation_report,
        ablation_rows=ablation_rows,
        bias_report=bias_report,
        evidence_report=evidence_report,
    )
    method_path = output_dir / "method_summary.md"
    method_path.write_text(method_summary, encoding="utf-8")
    data_availability = build_data_availability_statement(
        manifest_v2=manifest_v2,
        expansion_report=expansion_report,
        validation_report=validation_report,
        base_scores_path=base_scores_path,
        calibrated_results_path=calibrated_results_path,
    )
    data_availability_path = output_dir / "data_availability_statement.md"
    data_availability_path.write_text(data_availability, encoding="utf-8")

    report_lines = [
        "# BEA-Judge SCI Results Package",
        "",
        f"Created at: {utc_now()}",
        "",
        "## Generated Tables",
        "",
    ]
    for name, paths in tables.items():
        report_lines.append(f"- {name}: `{paths['md']}` and `{paths['csv']}`")
    report_lines.extend(
        [
            f"- method_summary: `{path_relative_to_root(method_path)}`",
            "",
            "## SCI Gates",
            "",
            markdown_table(
                [{"check": name, "passed": passed} for name, passed in gates["checks"].items()],
                ["check", "passed"],
            ),
        ]
    )
    report_path = output_dir / "sci_results_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    index = {
        "created_at": utc_now(),
        "inputs": {
            "base_scores": path_relative_to_root(base_scores_path),
            "repair_report": path_relative_to_root(repair_report_path),
            "validation_report": path_relative_to_root(validation_report_path),
            "calibrated_results": path_relative_to_root(calibrated_results_path),
            "ablation_report": path_relative_to_root(ablation_report_path),
            "bias_report": path_relative_to_root(bias_report_path),
            "bias_profiles": path_relative_to_root(bias_profiles_path),
            "evidence_report": path_relative_to_root(evidence_report_path),
            "evidence_profiles": path_relative_to_root(evidence_profiles_path),
            "swap_report": path_relative_to_root(swap_report_path) if swap_report else None,
            "expansion_report": path_relative_to_root(expansion_report_path) if expansion_report else None,
            "manifest_v2": path_relative_to_root(manifest_v2_path) if manifest_v2 else None,
            "calibration_comparison": path_relative_to_root(calibration_comparison_path)
            if calibration_summary and calibration_comparison_path
            else None,
        },
        "outputs": {
            **tables,
            "method_summary": path_relative_to_root(method_path),
            "data_availability_statement": path_relative_to_root(data_availability_path),
            "sci_results_report": path_relative_to_root(report_path),
        },
        "gates": gates,
    }
    index_path = output_dir / "sci_results_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BEA-Judge SCI result tables.")
    parser.add_argument("--base-scores", type=Path, default=DEFAULT_BASE_SCORES)
    parser.add_argument("--repair-report", type=Path, default=DEFAULT_REPAIR_REPORT)
    parser.add_argument("--validation-report", type=Path, default=MODEL_OUT / "latest_validation_report.json")
    parser.add_argument("--calibrated-results", type=Path, default=None)
    parser.add_argument("--ablation-report", type=Path, default=MODEL_OUT / "latest_ablation_report.json")
    parser.add_argument("--bias-report", type=Path, default=DATASETS / "bias_awareness_report.json")
    parser.add_argument("--bias-profiles", type=Path, default=DEFAULT_BIAS_PROFILES)
    parser.add_argument("--evidence-report", type=Path, default=DATASETS / "evidence_fact_report.json")
    parser.add_argument("--evidence-profiles", type=Path, default=DEFAULT_EVIDENCE_PROFILES)
    parser.add_argument("--swap-report", type=Path, default=DEFAULT_SWAP_REPORT)
    parser.add_argument("--expansion-report", type=Path, default=DEFAULT_EXPANSION_REPORT)
    parser.add_argument("--manifest-v2", type=Path, default=DEFAULT_MANIFEST_V2)
    parser.add_argument("--calibration-comparison", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs" / "experiment.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrated_results = args.calibrated_results or resolve_latest_calibrated_results(args.experiment_config)
    index = generate_outputs(
        base_scores_path=args.base_scores,
        repair_report_path=args.repair_report,
        validation_report_path=args.validation_report,
        calibrated_results_path=calibrated_results,
        ablation_report_path=args.ablation_report,
        bias_report_path=args.bias_report,
        bias_profiles_path=args.bias_profiles,
        evidence_report_path=args.evidence_report,
        evidence_profiles_path=args.evidence_profiles,
        output_dir=args.output_dir,
        swap_report_path=args.swap_report,
        expansion_report_path=args.expansion_report,
        manifest_v2_path=args.manifest_v2,
        calibration_comparison_path=args.calibration_comparison,
    )
    print(json.dumps(index["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
