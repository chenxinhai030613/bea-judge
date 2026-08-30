"""Bias-awareness utilities for BEA-Judge outputs."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PAIRWISE_LABELS = {"A>B", "B>A", "Tie"}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def favor_side(label: Any) -> Optional[str]:
    if label == "A>B":
        return "a"
    if label == "B>A":
        return "b"
    if label == "Tie":
        return "tie"
    return None


def bullet_count(text: Any) -> int:
    return len(re.findall(r"(^|\n)\s*[-*0-9]+[.)、-]", normalize_text(text)))


def safe_round(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 6)


def prediction_label(prediction: Optional[Dict[str, Any]]) -> Optional[str]:
    if not prediction:
        return None
    return prediction.get("predicted_label") or prediction.get("pairwise_label")


def longer_side(sample: Dict[str, Any], *, min_ratio: float = 1.15) -> Optional[str]:
    len_a = len(normalize_text(sample.get("answer_a")))
    len_b = len(normalize_text(sample.get("answer_b")))
    if not len_a or not len_b:
        return None
    ratio = max(len_a, len_b) / max(1, min(len_a, len_b))
    if ratio < min_ratio:
        return None
    return "a" if len_a > len_b else "b"


def length_bias_risk(sample: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    side = longer_side(sample)
    if side is None:
        return 0.0, reasons

    predicted_side = favor_side(prediction_label(prediction))
    gold_side = favor_side(sample.get("human_label"))
    if predicted_side == side and gold_side not in {side, "tie"}:
        reasons.append("prediction_favors_longer_answer_against_gold")
        return 1.0, reasons
    if predicted_side == side and gold_side == "tie":
        reasons.append("prediction_favors_longer_answer_on_gold_tie")
        return 0.7, reasons
    if predicted_side is None:
        reasons.append("length_imbalance_present_without_prediction")
        return 0.25, reasons
    return 0.0, reasons


def format_bias_risk(sample: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    meta = sample.get("metadata") or {}
    perturbation = normalize_text(meta.get("perturbation_applied") or meta.get("bias_type"))
    bullet_a = bullet_count(sample.get("answer_a"))
    bullet_b = bullet_count(sample.get("answer_b"))
    formatted_side = None
    if abs(bullet_a - bullet_b) >= 2:
        formatted_side = "a" if bullet_a > bullet_b else "b"

    predicted_side = favor_side(prediction_label(prediction))
    gold_side = favor_side(sample.get("human_label"))
    if perturbation == "format" and prediction and prediction_label(prediction) != sample.get("human_label"):
        reasons.append("format_perturbation_prediction_mismatch")
        return 1.0, reasons
    if formatted_side and predicted_side == formatted_side and gold_side not in {formatted_side, "tie"}:
        reasons.append("prediction_favors_more_formatted_answer_against_gold")
        return 0.8, reasons
    if perturbation == "format":
        reasons.append("format_perturbation_present")
        return 0.35, reasons
    return 0.0, reasons


def position_bias_risk(sample: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    meta = sample.get("metadata") or {}
    perturbation = normalize_text(meta.get("perturbation_applied") or meta.get("bias_type"))
    if perturbation != "position":
        return 0.0, reasons
    if prediction and prediction_label(prediction) != sample.get("human_label"):
        reasons.append("position_perturbation_prediction_mismatch")
        return 1.0, reasons
    reasons.append("position_perturbation_present")
    return 0.35, reasons


def rubric_bias_risk(sample: Dict[str, Any], prediction: Optional[Dict[str, Any]] = None) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    meta = sample.get("metadata") or {}
    perturbation = normalize_text(meta.get("perturbation_applied") or meta.get("bias_type"))
    if perturbation != "rubric_sensitivity":
        return 0.0, reasons
    if prediction and prediction_label(prediction) != sample.get("human_label"):
        reasons.append("rubric_sensitivity_prediction_mismatch")
        return 1.0, reasons
    reasons.append("rubric_sensitivity_present")
    return 0.35, reasons


def source_bias_risk(dataset_accuracy_gap: Optional[float]) -> float:
    if dataset_accuracy_gap is None:
        return 0.0
    return safe_round(abs(dataset_accuracy_gap))


def build_bias_profile(
    sample: Dict[str, Any],
    prediction: Optional[Dict[str, Any]] = None,
    *,
    dataset_accuracy_gap: Optional[float] = None,
    review_threshold: float = 0.5,
) -> Dict[str, Any]:
    position_risk, position_reasons = position_bias_risk(sample, prediction)
    length_risk, length_reasons = length_bias_risk(sample, prediction)
    format_risk, format_reasons = format_bias_risk(sample, prediction)
    rubric_risk, rubric_reasons = rubric_bias_risk(sample, prediction)
    source_risk = source_bias_risk(dataset_accuracy_gap)
    risks = [position_risk, length_risk, format_risk, rubric_risk, source_risk]
    overall = safe_round(max(risks) if risks else 0.0)
    reasons = position_reasons + length_reasons + format_reasons + rubric_reasons
    if source_risk:
        reasons.append("dataset_accuracy_gap")

    pred_label = prediction_label(prediction)
    confidence = prediction.get("confidence") if prediction else None
    return {
        "id": sample.get("id"),
        "dataset": sample.get("dataset"),
        "task_type": sample.get("task_type"),
        "split": sample.get("split"),
        "metadata": {
            "bias_type": (sample.get("metadata") or {}).get("bias_type", "none"),
            "perturbation_applied": (sample.get("metadata") or {}).get("perturbation_applied", "none"),
            "parent_id": (sample.get("metadata") or {}).get("parent_id"),
        },
        "prediction": {
            "predicted_label": pred_label,
            "gold_label": sample.get("human_label"),
            "confidence": confidence,
            "correct": pred_label == sample.get("human_label") if pred_label is not None else None,
        },
        "bias": {
            "position_risk": safe_round(position_risk),
            "length_risk": safe_round(length_risk),
            "format_risk": safe_round(format_risk),
            "rubric_sensitivity_risk": safe_round(rubric_risk),
            "source_bias_risk": safe_round(source_risk),
            "overall_bias_risk": overall,
            "review_required": overall >= review_threshold,
            "reasons": sorted(set(reasons)),
        },
    }


def _accuracy(rows: Iterable[Dict[str, Any]]) -> Optional[float]:
    correctness: List[bool] = []
    for row in rows:
        prediction = row.get("prediction", {})
        if prediction.get("correct") is not None:
            correctness.append(bool(prediction["correct"]))
            continue
        predicted = prediction.get("predicted_label")
        gold = prediction.get("gold_label")
        if predicted is not None and gold is not None:
            correctness.append(predicted == gold)
    if not correctness:
        return None
    return round(sum(1 for value in correctness if value) / len(correctness), 4)


def _bucket_summary(profiles: List[Dict[str, Any]], key_fn) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        buckets[str(key_fn(profile) or "none")].append(profile)
    summary: Dict[str, Dict[str, Any]] = {}
    for key, rows in sorted(buckets.items()):
        summary[key] = {
            "count": len(rows),
            "accuracy": _accuracy(rows),
            "review_rate": round(sum(1 for row in rows if row["bias"]["review_required"]) / len(rows), 4),
            "avg_overall_bias_risk": round(
                sum(float(row["bias"]["overall_bias_risk"]) for row in rows) / len(rows),
                4,
            ),
        }
    return summary


def summarize_bias_profiles(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not profiles:
        return {
            "overall": {"profile_count": 0, "accuracy": None, "review_rate": 0.0, "avg_overall_bias_risk": 0.0},
            "by_dataset": {},
            "by_bias_type": {},
        }
    review_rate = sum(1 for row in profiles if row["bias"]["review_required"]) / len(profiles)
    avg_risk = sum(float(row["bias"]["overall_bias_risk"]) for row in profiles) / len(profiles)
    reason_counts = Counter(reason for row in profiles for reason in row["bias"].get("reasons", []))
    return {
        "overall": {
            "profile_count": len(profiles),
            "accuracy": _accuracy(profiles),
            "review_rate": round(review_rate, 4),
            "avg_overall_bias_risk": round(avg_risk, 4),
        },
        "by_dataset": _bucket_summary(profiles, lambda row: row.get("dataset")),
        "by_task_type": _bucket_summary(profiles, lambda row: row.get("task_type")),
        "by_bias_type": _bucket_summary(profiles, lambda row: row.get("metadata", {}).get("bias_type")),
        "reason_counts": dict(reason_counts),
    }


def load_calibrated_predictions(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: Dict[str, Dict[str, Any]] = {}
    for split_payload in payload.values():
        if not isinstance(split_payload, dict):
            continue
        for head_rows in split_payload.values():
            if not isinstance(head_rows, list):
                continue
            for row in head_rows:
                if row.get("head") == "pairwise" and row.get("id"):
                    rows[str(row["id"])] = row
    return rows


def dataset_accuracy_gaps(samples: List[Dict[str, Any]], predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    rows_by_dataset: Dict[str, List[bool]] = defaultdict(list)
    all_correct: List[bool] = []
    for sample in samples:
        pred = predictions.get(str(sample.get("id")))
        if not pred:
            continue
        predicted = prediction_label(pred)
        gold = sample.get("human_label")
        if predicted not in PAIRWISE_LABELS or gold not in PAIRWISE_LABELS:
            continue
        correct = predicted == gold
        rows_by_dataset[str(sample.get("dataset"))].append(correct)
        all_correct.append(correct)
    if not all_correct:
        return {}
    overall = sum(all_correct) / len(all_correct)
    return {
        dataset: round((sum(items) / len(items)) - overall, 6)
        for dataset, items in rows_by_dataset.items()
        if items
    }
