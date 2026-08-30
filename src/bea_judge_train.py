from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from bias_awareness import build_bias_profile
from dataset_adapter import samples_from_payload
from evidence_features import evidence_feature_dict, factuality_decision_signal


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "datasets" / "processed"
MODEL_OUT = ROOT / "datasets" / "model_outputs"

PAIRWISE_LABELS = ["A>B", "B>A", "Tie"]
FACTUALITY_LABELS = ["supported", "unsupported", "ambiguous"]
SEED = 42
FACTUALITY_AUDIT_ONLY_EVIDENCE_FEATURES = {
    "evidence_low_support_anchor_sentence_ratio_a",
    "evidence_max_low_support_anchor_gap_a",
    "evidence_anchored_hallucination_severity_a",
    "evidence_low_support_anchor_sentence_ratio_b",
    "evidence_max_low_support_anchor_gap_b",
    "evidence_anchored_hallucination_severity_b",
}


@dataclass(frozen=True)
class HyperParams:
    learning_rate: float
    batch_size: int
    l2: float
    epochs: int


@dataclass(frozen=True)
class JudgeOutputFeatures:
    rows: Dict[str, Dict[str, Any]]
    source_counts: Dict[str, int]
    path: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_dataset(path: Path) -> List[Dict[str, Any]]:
    return samples_from_payload(json.loads(path.read_text(encoding="utf-8")))


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", text.lower()))


def overlap(a: str, b: str) -> float:
    left = token_set(a)
    right = token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def sentence_count(text: str) -> int:
    return max(1, len([s for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]))


def bullet_count(text: str) -> int:
    return len(re.findall(r"(^|\n)\s*[-*0-9]+[.)、-]", text))


def numeric_count(text: str) -> int:
    return len(re.findall(r"\d+(?:\.\d+)?", text))


def safe_ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def load_judge_output_features(path: Path, *, allow_heuristic: bool = False) -> JudgeOutputFeatures:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"judge output must be a base_scores.json list, got {type(payload).__name__}: {path}")
    rows: Dict[str, Dict[str, Any]] = {}
    source_counts: Dict[str, int] = {
        "prometheus2": 0,
        "m_prometheus": 0,
        "m_prometheus_qlora": 0,
        "heuristic_fallback": 0,
        "backend_error": 0,
        "other": 0,
    }
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(f"judge output rows must be objects: {path}")
        backend = str(row.get("judge_backend", "other"))
        parse_status = str(row.get("parse_status", ""))
        if parse_status == "backend_error":
            source_counts["backend_error"] += 1
            continue
        if backend in {"prometheus2", "m_prometheus", "m_prometheus_qlora"}:
            source_counts[backend] += 1
        elif backend == "heuristic_fallback":
            source_counts["heuristic_fallback"] += 1
            if not allow_heuristic:
                continue
        else:
            source_counts["other"] += 1
            continue

        scores = row.get("parsed_scores", {})
        score_a = scores.get("score_a")
        score_b = scores.get("score_b")
        pred_label = row.get("pred_label")
        sample_id = row.get("id")
        if sample_id is None or pred_label not in PAIRWISE_LABELS or score_a is None or score_b is None:
            source_counts["backend_error"] += 1
            continue
        rows[str(sample_id)] = {
            "pred_label": pred_label,
            "score_a": float(score_a),
            "score_b": float(score_b),
            "judge_backend": backend,
            "swap_available": float(row.get("swap_available", 0.0) or 0.0),
            "swap_consistency_flag": float(row.get("swap_consistency_flag", 0.0) or 0.0),
            "swap_margin_delta": float(row.get("swap_margin_delta", 0.0) or 0.0),
        }
    return JudgeOutputFeatures(rows=rows, source_counts=source_counts, path=str(path))


def validate_judge_output_coverage(
    samples: Sequence[Dict[str, Any]],
    judge_outputs: JudgeOutputFeatures,
    labels: Sequence[str] = PAIRWISE_LABELS,
) -> Dict[str, Any]:
    required_ids = [str(sample.get("id")) for sample in select_samples(samples, labels)]
    missing = [sample_id for sample_id in required_ids if sample_id not in judge_outputs.rows]
    report = {
        "required_pairwise_rows": len(required_ids),
        "covered_pairwise_rows": len(required_ids) - len(missing),
        "missing_pairwise_rows": len(missing),
        "missing_examples": missing[:10],
        "judge_output_path": judge_outputs.path,
    }
    if missing:
        raise ValueError(
            "base judge coverage check failed: "
            f"missing {len(missing)} of {len(required_ids)} required pairwise base judge rows; "
            f"examples={missing[:10]}"
        )
    return report


def base_pairwise_features(sample: Dict[str, Any], judge_outputs: JudgeOutputFeatures) -> Dict[str, float]:
    sample_id = str(sample.get("id"))
    row = judge_outputs.rows[sample_id]
    pred_label = str(row["pred_label"])
    score_a = float(row["score_a"])
    score_b = float(row["score_b"])
    diff = score_a - score_b
    return {
        "base_score_a": score_a,
        "base_score_b": score_b,
        "base_score_diff": diff,
        "base_margin": abs(diff),
        "base_pred_a": 1.0 if pred_label == "A>B" else 0.0,
        "base_pred_b": 1.0 if pred_label == "B>A" else 0.0,
        "base_pred_tie": 1.0 if pred_label == "Tie" else 0.0,
        "swap_available": float(row.get("swap_available", 0.0)),
        "swap_consistency_flag": float(row.get("swap_consistency_flag", 0.0)),
        "swap_margin_delta": float(row.get("swap_margin_delta", 0.0)),
    }


def text_pairwise_features(sample: Dict[str, Any]) -> Dict[str, float]:
    prompt = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference"))
    answer_a = normalize_text(sample.get("answer_a"))
    answer_b = normalize_text(sample.get("answer_b"))
    len_a = len(answer_a)
    len_b = len(answer_b)
    sent_a = sentence_count(answer_a)
    sent_b = sentence_count(answer_b)
    bullets_a = bullet_count(answer_a)
    bullets_b = bullet_count(answer_b)
    nums_a = numeric_count(answer_a)
    nums_b = numeric_count(answer_b)

    return {
        "prompt_chars": len(prompt),
        "context_chars": len(context),
        "reference_chars": len(reference),
        "answer_a_chars": len_a,
        "answer_b_chars": len_b,
        "answer_len_diff": len_a - len_b,
        "answer_len_ratio": safe_ratio(len_a, max(1, len_b)),
        "sentence_diff": sent_a - sent_b,
        "bullet_diff": bullets_a - bullets_b,
        "numeric_diff": nums_a - nums_b,
        "prompt_overlap_a": overlap(prompt, answer_a),
        "prompt_overlap_b": overlap(prompt, answer_b),
        "prompt_overlap_diff": overlap(prompt, answer_a) - overlap(prompt, answer_b),
        "context_overlap_a": overlap(context, answer_a),
        "context_overlap_b": overlap(context, answer_b),
        "context_overlap_diff": overlap(context, answer_a) - overlap(context, answer_b),
        "reference_overlap_a": overlap(reference, answer_a),
        "reference_overlap_b": overlap(reference, answer_b),
        "reference_overlap_diff": overlap(reference, answer_a) - overlap(reference, answer_b),
    }


def bias_features(sample: Dict[str, Any]) -> Dict[str, float]:
    meta = sample.get("metadata", {})
    bias_type = normalize_text(meta.get("bias_type"))
    perturbation = normalize_text(meta.get("perturbation_applied"))
    return {
        "bias_position": 1.0 if bias_type == "position" or perturbation == "position" else 0.0,
        "bias_length": 1.0 if bias_type == "length" or perturbation == "length" else 0.0,
        "bias_format": 1.0 if bias_type == "format" or perturbation == "format" else 0.0,
        "bias_rubric": 1.0 if bias_type == "rubric_sensitivity" or perturbation == "rubric_sensitivity" else 0.0,
        "is_synthetic_perturbed": 1.0 if sample.get("dataset") == "synthetic_perturbed" else 0.0,
    }


def bias_risk_features(sample: Dict[str, Any]) -> Dict[str, float]:
    profile = build_bias_profile(sample)
    bias = profile["bias"]
    return {
        "bias_position_risk": float(bias["position_risk"]),
        "bias_length_risk": float(bias["length_risk"]),
        "bias_format_risk": float(bias["format_risk"]),
        "bias_rubric_sensitivity_risk": float(bias["rubric_sensitivity_risk"]),
        "bias_source_risk": float(bias["source_bias_risk"]),
        "bias_overall_risk": float(bias["overall_bias_risk"]),
        "bias_review_required": 1.0 if bias["review_required"] else 0.0,
    }


def factuality_text_features(sample: Dict[str, Any]) -> Dict[str, float]:
    prompt = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference"))
    answer = normalize_text(sample.get("answer_a"))
    return {
        "prompt_chars": len(prompt),
        "context_chars": len(context),
        "reference_chars": len(reference),
        "answer_chars": len(answer),
        "answer_sentences": sentence_count(answer),
        "answer_numbers": numeric_count(answer),
        "answer_context_overlap": overlap(answer, context),
        "answer_reference_overlap": overlap(answer, reference),
        "prompt_context_overlap": overlap(prompt, context),
        "reference_context_overlap": overlap(reference, context),
        "has_reference": 1.0 if reference else 0.0,
        "is_single_answer": 1.0
        if sample.get("metadata", {}).get("factuality_task_form") == "single_answer"
        else 0.0,
    }


def one_hot_features(sample: Dict[str, Any]) -> Dict[str, float]:
    dataset = normalize_text(sample.get("dataset"))
    task = normalize_text(sample.get("task_type"))
    scoring = normalize_text(sample.get("metadata", {}).get("scoring_system"))
    return {
        f"dataset={dataset}": 1.0,
        f"task={task}": 1.0,
        f"scoring={scoring}": 1.0,
    }


def pairwise_feature_dict(sample: Dict[str, Any], judge_outputs: JudgeOutputFeatures) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(text_pairwise_features(sample))
    features.update(base_pairwise_features(sample, judge_outputs))
    features.update(bias_features(sample))
    features.update(bias_risk_features(sample))
    features.update(one_hot_features(sample))
    if factuality_decision_signal(sample) == "pairwise_factuality":
        features.update(evidence_feature_dict(sample))
    return features


def factuality_feature_dict(sample: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(factuality_text_features(sample))
    features.update(
        {
            key: value
            for key, value in evidence_feature_dict(sample).items()
            if key not in FACTUALITY_AUDIT_ONLY_EVIDENCE_FEATURES
        }
    )
    features.update(one_hot_features(sample))
    return features


def make_matrix(
    samples: Sequence[Dict[str, Any]],
    feature_fn,
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    dicts = [feature_fn(sample) for sample in samples]
    if feature_names is None:
        names = sorted({name for row in dicts for name in row})
    else:
        names = feature_names
    matrix = np.zeros((len(samples), len(names)), dtype=float)
    for i, row in enumerate(dicts):
        for j, name in enumerate(names):
            matrix[i, j] = float(row.get(name, 0.0))
    return matrix, names


def standardize_train_dev(
    x_train: np.ndarray,
    x_dev: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (x_train - mean) / std, (x_dev - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def apply_scaler(x: np.ndarray, scaler: Dict[str, List[float]]) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=float)
    std = np.array(scaler["std"], dtype=float)
    std[std < 1e-8] = 1.0
    return (x - mean) / std


class SoftmaxClassifier:
    def __init__(self, num_features: int, num_classes: int, seed: int = SEED) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.01, size=(num_features, num_classes))
        self.bias = np.zeros(num_classes)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.softmax(x @ self.weights + self.bias)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        params: HyperParams,
        seed: int = SEED,
        sample_weight: Optional[np.ndarray] = None,
    ) -> List[float]:
        rng = np.random.default_rng(seed)
        losses: List[float] = []
        n = x.shape[0]
        eye = np.eye(self.bias.shape[0])
        weights = np.ones(n, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape[0] != n:
            raise ValueError("sample_weight length must match x rows")
        weights = np.maximum(weights, 0.0)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(n, dtype=float)
        for _epoch in range(params.epochs):
            order = rng.permutation(n)
            for start in range(0, n, params.batch_size):
                idx = order[start : start + params.batch_size]
                xb = x[idx]
                yb = y[idx]
                wb = weights[idx]
                weight_sum = max(float(wb.sum()), 1e-12)
                probs = self.predict_proba(xb)
                target = eye[yb]
                grad_logits = ((probs - target) * wb[:, None]) / weight_sum
                grad_w = xb.T @ grad_logits + params.l2 * self.weights
                grad_b = grad_logits.sum(axis=0)
                self.weights -= params.learning_rate * grad_w
                self.bias -= params.learning_rate * grad_b
            probs_all = self.predict_proba(x)
            loss = float(np.average(-np.log(probs_all[np.arange(n), y] + 1e-12), weights=weights))
            loss += 0.5 * params.l2 * float((self.weights**2).sum())
            losses.append(round(float(loss), 6))
        return losses

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights.tolist(), "bias": self.bias.tolist()}


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    clipped = np.clip(probs, 1e-12, 1.0)
    logits = np.log(clipped) / temperature
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def negative_log_likelihood(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(-np.log(probs[np.arange(len(y_true)), y_true] + 1e-12).mean())


def calibrate_temperature(
    probs: np.ndarray,
    y_true: np.ndarray,
    candidates: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if candidates is None:
        candidates = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    rows: List[Dict[str, float]] = []
    best: Optional[Tuple[float, float]] = None
    for value in candidates:
        scaled = temperature_scale(probs, float(value))
        nll = negative_log_likelihood(y_true, scaled)
        ece = ece_score(y_true, scaled)
        brier = brier_score(y_true, scaled, probs.shape[1])
        objective = nll + ece + 0.25 * brier
        row = {
            "temperature": float(value),
            "nll": round(nll, 6),
            "ece": ece,
            "brier": brier,
            "objective": round(float(objective), 6),
        }
        rows.append(row)
        if best is None or objective < best[0]:
            best = (objective, float(value))
    assert best is not None
    return {
        "method": "temperature_scaling_grid_search",
        "selection_metric": "dev_nll + dev_ece + 0.25*dev_brier",
        "temperature": best[1],
        "candidates": rows,
    }


def select_review_threshold(
    risk: np.ndarray,
    correct: np.ndarray,
    target_error_recall: float = 0.80,
) -> Dict[str, Any]:
    if len(risk) == 0:
        return {
            "method": "risk_threshold_on_dev",
            "risk_signal": "1 - confidence",
            "threshold": 0.0,
            "target_error_recall": target_error_recall,
            "error_recall": 0.0,
            "review_rate": 0.0,
            "error_count": 0,
            "review_count": 0,
        }
    errors = ~correct.astype(bool)
    error_count = int(errors.sum())
    candidates = sorted({float(x) for x in risk})
    if error_count == 0:
        threshold = candidates[0]
        reviewed = risk >= threshold
        return {
            "method": "risk_threshold_on_dev",
            "risk_signal": "1 - confidence",
            "threshold": round(float(threshold), 6),
            "target_error_recall": target_error_recall,
            "error_recall": 0.0,
            "review_rate": round(float(reviewed.mean()), 4),
            "error_count": 0,
            "review_count": int(reviewed.sum()),
        }

    best: Optional[Dict[str, Any]] = None
    for threshold in candidates:
        reviewed = risk >= threshold
        captured_errors = int((reviewed & errors).sum())
        error_recall = captured_errors / error_count
        review_rate = float(reviewed.mean())
        row = {
            "method": "risk_threshold_on_dev",
            "risk_signal": "1 - confidence",
            "threshold": round(float(threshold), 6),
            "target_error_recall": target_error_recall,
            "error_recall": round(float(error_recall), 4),
            "review_rate": round(review_rate, 4),
            "error_count": error_count,
            "review_count": int(reviewed.sum()),
        }
        if error_recall >= target_error_recall:
            if best is None or row["review_rate"] < best["review_rate"]:
                best = row
    if best is not None:
        return best

    threshold = candidates[-1]
    reviewed = risk >= threshold
    return {
        "method": "risk_threshold_on_dev",
        "risk_signal": "1 - confidence",
        "threshold": round(float(threshold), 6),
        "target_error_recall": target_error_recall,
        "error_recall": 1.0,
        "review_rate": round(float(reviewed.mean()), 4),
        "error_count": error_count,
        "review_count": int(reviewed.sum()),
    }


def final_score_from_label(label: str, head_name: str) -> float:
    if head_name == "factuality":
        return {"supported": 1.0, "ambiguous": 0.5, "unsupported": 0.0}.get(label, 0.5)
    return {"A>B": 1.0, "Tie": 0.5, "B>A": 0.0}.get(label, 0.5)


def risk_scores(confidence: np.ndarray) -> np.ndarray:
    return 1.0 - confidence


def bias_review_reason(sample: Dict[str, Any], head_name: str) -> Optional[str]:
    if head_name != "pairwise":
        return None
    bias = build_bias_profile(sample)["bias"]
    if float(bias["position_risk"]) >= 0.5:
        return "bias_position_high_risk"
    if float(bias["rubric_sensitivity_risk"]) >= 0.5:
        return "bias_rubric_high_risk"
    return None


def validate_calibrated_rows_schema(rows: Sequence[Dict[str, Any]]) -> None:
    required = {
        "id",
        "dataset",
        "task_type",
        "split",
        "head",
        "human_label",
        "predicted_label",
        "final_score",
        "confidence",
        "risk_score",
        "review_flag",
        "review_reason",
        "label_probabilities",
    }
    for i, row in enumerate(rows):
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"calibrated row {i} missing required fields: {missing}")
        if not isinstance(row["label_probabilities"], dict) or not row["label_probabilities"]:
            raise ValueError(f"calibrated row {i} must contain non-empty label_probabilities")
        confidence = float(row["confidence"])
        risk = float(row["risk_score"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"calibrated row {i} confidence outside [0, 1]")
        if not 0.0 <= risk <= 1.0:
            raise ValueError(f"calibrated row {i} risk_score outside [0, 1]")


def make_calibrated_rows(
    samples: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    probs: np.ndarray,
    review_threshold: float,
    head_name: str,
    pred_indices: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    pred = probs.argmax(axis=1) if pred_indices is None else pred_indices
    confidence = confidence_for_predictions(probs, pred)
    for sample, pred_index, conf, row_probs in zip(samples, pred, confidence, probs):
        label = labels[int(pred_index)]
        risk = 1.0 - float(conf)
        reasons: List[str] = []
        if risk >= review_threshold:
            reasons.append("risk_threshold")
        bias_reason = bias_review_reason(sample, head_name)
        if bias_reason:
            reasons.append(bias_reason)
        review_flag = bool(reasons)
        rows.append(
            {
                "id": sample.get("id"),
                "dataset": sample.get("dataset"),
                "task_type": sample.get("task_type"),
                "split": sample.get("split"),
                "head": head_name,
                "human_label": sample.get("human_label"),
                "predicted_label": label,
                "pairwise_label": label if head_name == "pairwise" else None,
                "factuality_label": label if head_name == "factuality" else None,
                "final_score": round(final_score_from_label(label, head_name), 4),
                "confidence": round(float(conf), 6),
                "risk_score": round(risk, 6),
                "review_flag": review_flag,
                "review_reason": "+".join(reasons) if reasons else "auto_accept",
                "label_probabilities": {
                    label_name: round(float(prob), 6)
                    for label_name, prob in zip(labels, row_probs)
                },
            }
        )
    validate_calibrated_rows_schema(rows)
    return rows


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return round(float((y_true == y_pred).mean()), 4) if len(y_true) else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[int]) -> float:
    scores: List[float] = []
    for label in labels:
        if not (y_true == label).any() and not (y_pred == label).any():
            continue
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(score)
    return round(float(sum(scores) / len(scores)), 4) if scores else 0.0


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0:
        return 0.0
    pred = probs.argmax(axis=1)
    conf = confidence_for_predictions(probs, pred)
    return ece_score_from_predictions(y_true, pred, conf, bins=bins)


def confidence_for_predictions(probs: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if len(pred) == 0:
        return np.array([], dtype=float)
    return probs[np.arange(len(pred)), pred]


def ece_score_from_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    confidence: np.ndarray,
    bins: int = 10,
) -> float:
    if len(y_true) == 0:
        return 0.0
    total = len(y_true)
    ece = 0.0
    for b in range(bins):
        lo = b / bins
        hi = (b + 1) / bins
        mask = (confidence >= lo) & (confidence < hi if b < bins - 1 else confidence <= hi)
        if not mask.any():
            continue
        acc = (pred[mask] == y_true[mask]).mean()
        avg_conf = confidence[mask].mean()
        ece += float(mask.sum()) / total * abs(float(acc) - float(avg_conf))
    return round(ece, 4)


def brier_score(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    target = np.eye(num_classes)[y_true]
    return round(float(((probs - target) ** 2).sum(axis=1).mean()), 4) if len(y_true) else 0.0


def confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {label: {p: 0 for p in labels} for label in labels}
    for actual, pred in zip(y_true, y_pred):
        out[labels[int(actual)]][labels[int(pred)]] += 1
    return out


def tie_recall(y_true: np.ndarray, pred: np.ndarray, labels: Sequence[str]) -> Optional[float]:
    if "Tie" not in labels:
        return None
    tie_index = list(labels).index("Tie")
    mask = y_true == tie_index
    if not mask.any():
        return None
    return round(float((pred[mask] == tie_index).mean()), 4)


def disabled_tie_policy() -> Dict[str, Any]:
    return {
        "enabled": False,
        "method": "disabled",
        "min_tie_probability": None,
        "max_ab_margin": None,
        "max_base_margin": None,
        "dataset_policies": {},
        "selection_metric": "not_applicable",
        "candidates": [],
    }


def apply_pairwise_tie_policy(
    probs: np.ndarray,
    labels: Sequence[str],
    policy: Optional[Dict[str, Any]],
    base_margins: Optional[np.ndarray] = None,
    datasets: Optional[Sequence[str]] = None,
) -> np.ndarray:
    pred = probs.argmax(axis=1)
    if list(labels) != PAIRWISE_LABELS or not policy or not policy.get("enabled"):
        return pred
    a_index = labels.index("A>B")
    b_index = labels.index("B>A")
    tie_index = labels.index("Tie")
    adjusted = pred.copy()

    def tie_mask_for(row_policy: Dict[str, Any]) -> np.ndarray:
        min_tie_probability = row_policy.get("min_tie_probability")
        max_ab_margin = row_policy.get("max_ab_margin")
        if min_tie_probability is None or max_ab_margin is None:
            return np.zeros(len(probs), dtype=bool)
        mask = (
            (probs[:, tie_index] >= float(min_tie_probability))
            & (np.abs(probs[:, a_index] - probs[:, b_index]) <= float(max_ab_margin))
        )
        max_base_margin = row_policy.get("max_base_margin")
        if max_base_margin is not None and base_margins is not None:
            mask = mask & (base_margins <= float(max_base_margin))
        return mask

    adjusted[tie_mask_for(policy)] = tie_index
    dataset_policies = policy.get("dataset_policies") or {}
    if datasets is not None and dataset_policies:
        dataset_array = np.array(list(datasets), dtype=object)
        for dataset, row_policy in dataset_policies.items():
            mask = tie_mask_for(row_policy) & (dataset_array == dataset)
            adjusted[mask] = tie_index
    return adjusted


def disabled_factuality_threshold_policy() -> Dict[str, Any]:
    return {
        "enabled": False,
        "method": "disabled",
        "dataset": None,
        "unsupported_threshold": None,
        "selection_metric": "not_applicable",
        "candidates": [],
    }


def apply_factuality_threshold_policy(
    probs: np.ndarray,
    labels: Sequence[str],
    datasets: Sequence[str],
    policy: Optional[Dict[str, Any]],
) -> np.ndarray:
    pred = probs.argmax(axis=1)
    if (
        not policy
        or not policy.get("enabled")
        or "supported" not in labels
        or "unsupported" not in labels
    ):
        return pred
    dataset_name = str(policy.get("dataset") or "ragtruth")
    threshold = float(policy["unsupported_threshold"])
    supported_index = labels.index("supported")
    unsupported_index = labels.index("unsupported")
    adjusted = pred.copy()
    for i, dataset in enumerate(datasets):
        if dataset != dataset_name:
            continue
        adjusted[i] = unsupported_index if probs[i, unsupported_index] >= threshold else supported_index
    return adjusted


def metrics_from_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
    labels: Sequence[str],
) -> Dict[str, Any]:
    confidence = confidence_for_predictions(probs, pred)
    return {
        "accuracy": accuracy(y_true, pred),
        "macro_f1": macro_f1(y_true, pred, list(range(len(labels)))),
        "ece": ece_score_from_predictions(y_true, pred, confidence),
        "brier": brier_score(y_true, probs, len(labels)),
        "tie_recall": tie_recall(y_true, pred, labels),
        "confusion": confusion(y_true, pred, labels),
        "pred_distribution": dict(Counter(labels[int(x)] for x in pred)),
        "gold_distribution": dict(Counter(labels[int(x)] for x in y_true)),
    }


def select_factuality_threshold_policy(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    datasets: Sequence[str],
    dataset_name: str = "ragtruth",
) -> Dict[str, Any]:
    if "supported" not in labels or "unsupported" not in labels:
        return disabled_factuality_threshold_policy()
    indices = [i for i, dataset in enumerate(datasets) if dataset == dataset_name]
    if len(indices) < 10:
        policy = disabled_factuality_threshold_policy()
        policy["method"] = "insufficient_dataset_rows"
        policy["dataset"] = dataset_name
        return policy
    baseline_pred = probs.argmax(axis=1)
    baseline_metrics = metrics_from_predictions(y_true, probs, baseline_pred, labels)
    rows: List[Dict[str, Any]] = [
        {
            "enabled": False,
            "dataset": dataset_name,
            "unsupported_threshold": None,
            "objective": round(
                float(baseline_metrics["macro_f1"] + 0.25 * baseline_metrics["accuracy"] - 0.05 * baseline_metrics["ece"]),
                6,
            ),
            "metrics": baseline_metrics,
        }
    ]
    for threshold in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        policy = {
            "enabled": True,
            "dataset": dataset_name,
            "unsupported_threshold": threshold,
        }
        pred = apply_factuality_threshold_policy(probs, labels, datasets, policy)
        candidate_metrics = metrics_from_predictions(y_true, probs, pred, labels)
        objective = candidate_metrics["macro_f1"] + 0.25 * candidate_metrics["accuracy"] - 0.05 * candidate_metrics["ece"]
        rows.append(
            {
                "enabled": True,
                "dataset": dataset_name,
                "unsupported_threshold": threshold,
                "objective": round(float(objective), 6),
                "metrics": candidate_metrics,
            }
        )
    best = max(
        rows,
        key=lambda row: (
            row["metrics"]["macro_f1"],
            row["metrics"]["accuracy"],
            -row["metrics"]["ece"],
            row["objective"],
        ),
    )
    enabled = bool(best["enabled"]) and best["metrics"]["macro_f1"] >= baseline_metrics["macro_f1"]
    return {
        "enabled": enabled,
        "method": "dev_ragtruth_unsupported_threshold",
        "dataset": dataset_name,
        "unsupported_threshold": best["unsupported_threshold"] if enabled else None,
        "selection_metric": "macro_f1_first_then_accuracy_minus_ece",
        "baseline_metrics": baseline_metrics,
        "selected_objective": best["objective"],
        "selected_metrics": best["metrics"] if enabled else baseline_metrics,
        "candidates": sorted(rows, key=lambda row: row["objective"], reverse=True),
    }


def select_pairwise_tie_policy(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    base_margins: Optional[np.ndarray] = None,
    datasets: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if list(labels) != PAIRWISE_LABELS:
        return disabled_tie_policy()

    rows: List[Dict[str, Any]] = []
    baseline_pred = probs.argmax(axis=1)
    baseline_metrics = metrics_from_predictions(y_true, probs, baseline_pred, labels)
    baseline_objective = (
        baseline_metrics["macro_f1"]
        + 0.25 * baseline_metrics["accuracy"]
        + 0.10 * float(baseline_metrics.get("tie_recall") or 0.0)
        - 0.05 * baseline_metrics["ece"]
    )
    rows.append(
        {
            "enabled": False,
            "min_tie_probability": None,
            "max_ab_margin": None,
            "max_base_margin": None,
            "dataset_policies": {},
            "objective": round(float(baseline_objective), 6),
            "metrics": baseline_metrics,
        }
    )

    base_margin_candidates: Sequence[Optional[float]]
    base_margin_candidates = (None, 0.05, 0.10, 0.25, 0.50, 1.00) if base_margins is not None else (None,)
    for min_tie_probability in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        for max_ab_margin in (0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.00):
            for max_base_margin in base_margin_candidates:
                policy = {
                    "enabled": True,
                    "min_tie_probability": min_tie_probability,
                    "max_ab_margin": max_ab_margin,
                    "max_base_margin": max_base_margin,
                    "dataset_policies": {},
                }
                pred = apply_pairwise_tie_policy(probs, labels, policy, base_margins=base_margins)
                candidate_metrics = metrics_from_predictions(y_true, probs, pred, labels)
                objective = (
                    candidate_metrics["macro_f1"]
                    + 0.25 * candidate_metrics["accuracy"]
                    + 0.10 * float(candidate_metrics.get("tie_recall") or 0.0)
                    - 0.05 * candidate_metrics["ece"]
                )
                rows.append(
                    {
                        "enabled": True,
                        "min_tie_probability": min_tie_probability,
                        "max_ab_margin": max_ab_margin,
                        "max_base_margin": max_base_margin,
                        "objective": round(float(objective), 6),
                        "metrics": candidate_metrics,
                    }
                )

    macro_floor = baseline_metrics["macro_f1"] - 0.005
    accuracy_floor = baseline_metrics["accuracy"] - 0.010
    ece_ceiling = baseline_metrics["ece"] + 0.020
    baseline_tie = float(baseline_metrics.get("tie_recall") or 0.0)
    eligible = [
        row
        for row in rows
        if row["metrics"]["macro_f1"] >= macro_floor
        and row["metrics"]["accuracy"] >= accuracy_floor
        and row["metrics"]["ece"] <= ece_ceiling
        and (
            float(row["metrics"].get("tie_recall") or 0.0) > baseline_tie
            if row["enabled"]
            else float(row["metrics"].get("tie_recall") or 0.0) >= baseline_tie
        )
    ]
    enabled_eligible = [row for row in eligible if row["enabled"]]
    if enabled_eligible:
        best = max(
            enabled_eligible,
            key=lambda row: (
                float(row["metrics"].get("tie_recall") or 0.0),
                row["metrics"]["macro_f1"],
                row["metrics"]["accuracy"],
                -row["metrics"]["ece"],
                row["objective"],
            ),
        )
        selection_metric = "tie_recall_first_with_dev_quality_floors"
    else:
        best = max(rows, key=lambda row: row["objective"])
        selection_metric = "macro_f1 + 0.25*accuracy + 0.10*tie_recall - 0.05*ece"

    if datasets is not None:
        dataset_policy = select_dataset_aware_tie_policy(
            probs,
            y_true,
            labels,
            datasets,
            base_margins=base_margins,
            macro_floor=macro_floor,
            accuracy_floor=accuracy_floor,
            ece_ceiling=ece_ceiling,
            baseline_tie=baseline_tie,
        )
        if dataset_policy is not None:
            rows.append(dataset_policy)
            if (
                float(dataset_policy["metrics"].get("tie_recall") or 0.0)
                > float(best["metrics"].get("tie_recall") or 0.0)
                and dataset_policy["metrics"]["macro_f1"] >= macro_floor
                and dataset_policy["metrics"]["accuracy"] >= accuracy_floor
                and dataset_policy["metrics"]["ece"] <= ece_ceiling
            ):
                best = dataset_policy
                selection_metric = "dataset_aware_tie_recall_with_dev_quality_floors"
        overlay_policy = select_dataset_overlay_tie_policy(
            probs,
            y_true,
            labels,
            datasets,
            base_policy=best,
            base_margins=base_margins,
            macro_floor=macro_floor,
            accuracy_floor=baseline_metrics["accuracy"] - 0.015,
            ece_ceiling=ece_ceiling,
            baseline_tie=float(best["metrics"].get("tie_recall") or 0.0),
        )
        if overlay_policy is not None:
            rows.append(overlay_policy)
            if (
                float(overlay_policy["metrics"].get("tie_recall") or 0.0)
                > float(best["metrics"].get("tie_recall") or 0.0)
                and overlay_policy["metrics"]["macro_f1"] >= macro_floor
                and overlay_policy["metrics"]["accuracy"] >= baseline_metrics["accuracy"] - 0.015
                and overlay_policy["metrics"]["ece"] <= ece_ceiling
            ):
                best = overlay_policy
                selection_metric = "dataset_overlay_tie_recall_with_dev_quality_floors"
    return {
        "enabled": bool(best["enabled"]),
        "method": "dev_probability_margin_tie_policy",
        "min_tie_probability": best["min_tie_probability"],
        "max_ab_margin": best["max_ab_margin"],
        "max_base_margin": best.get("max_base_margin"),
        "dataset_policies": best.get("dataset_policies", {}),
        "selection_metric": selection_metric,
        "selection_constraints": {
            "macro_f1_floor": round(float(macro_floor), 6),
            "accuracy_floor": round(float(accuracy_floor), 6),
            "ece_ceiling": round(float(ece_ceiling), 6),
            "baseline_tie_recall": round(float(baseline_tie), 6),
        },
        "selected_objective": best["objective"],
        "selected_metrics": best["metrics"],
        "candidates": sorted(rows, key=lambda row: row["objective"], reverse=True),
    }


def select_dataset_aware_tie_policy(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    datasets: Sequence[str],
    *,
    base_margins: Optional[np.ndarray],
    macro_floor: float,
    accuracy_floor: float,
    ece_ceiling: float,
    baseline_tie: float,
) -> Optional[Dict[str, Any]]:
    dataset_policies: Dict[str, Dict[str, Any]] = {}
    dataset_values = np.array(list(datasets), dtype=object)
    tie_index = labels.index("Tie")
    base_margin_candidates: Sequence[Optional[float]]
    base_margin_candidates = (None, 0.05, 0.10, 0.25, 0.50, 1.00) if base_margins is not None else (None,)
    for dataset in sorted(set(datasets)):
        indices = np.where(dataset_values == dataset)[0]
        if len(indices) < 20 or int((y_true[indices] == tie_index).sum()) < 3:
            continue
        subset_probs = probs[indices]
        subset_y = y_true[indices]
        subset_margins = base_margins[indices] if base_margins is not None else None
        baseline_pred = subset_probs.argmax(axis=1)
        baseline_metrics = metrics_from_predictions(subset_y, subset_probs, baseline_pred, labels)
        best: Optional[Dict[str, Any]] = None
        for min_tie_probability in (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35):
            for max_ab_margin in (0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50):
                for max_base_margin in base_margin_candidates:
                    policy = {
                        "enabled": True,
                        "min_tie_probability": min_tie_probability,
                        "max_ab_margin": max_ab_margin,
                        "max_base_margin": max_base_margin,
                        "dataset_policies": {},
                    }
                    pred = apply_pairwise_tie_policy(
                        subset_probs,
                        labels,
                        policy,
                        base_margins=subset_margins,
                    )
                    metric = metrics_from_predictions(subset_y, subset_probs, pred, labels)
                    if float(metric.get("tie_recall") or 0.0) <= float(baseline_metrics.get("tie_recall") or 0.0):
                        continue
                    if metric["macro_f1"] < baseline_metrics["macro_f1"] - 0.010:
                        continue
                    if metric["accuracy"] < baseline_metrics["accuracy"] - 0.020:
                        continue
                    if metric["ece"] > baseline_metrics["ece"] + 0.010:
                        continue
                    objective = (
                        metric["macro_f1"]
                        + 0.25 * metric["accuracy"]
                        + 0.20 * float(metric.get("tie_recall") or 0.0)
                        - 0.05 * metric["ece"]
                    )
                    row = {**policy, "objective": round(float(objective), 6), "metrics": metric}
                    if best is None or (
                        float(metric.get("tie_recall") or 0.0),
                        metric["macro_f1"],
                        metric["accuracy"],
                        -metric["ece"],
                        objective,
                    ) > (
                        float(best["metrics"].get("tie_recall") or 0.0),
                        best["metrics"]["macro_f1"],
                        best["metrics"]["accuracy"],
                        -best["metrics"]["ece"],
                        float(best["objective"]),
                    ):
                        best = row
        if best is not None:
            dataset_policies[dataset] = {
                "min_tie_probability": best["min_tie_probability"],
                "max_ab_margin": best["max_ab_margin"],
                "max_base_margin": best.get("max_base_margin"),
                "dev_subset_metrics": best["metrics"],
            }
    if not dataset_policies:
        return None

    policy = {
        "enabled": True,
        "min_tie_probability": None,
        "max_ab_margin": None,
        "max_base_margin": None,
        "dataset_policies": dataset_policies,
    }
    pred = apply_pairwise_tie_policy(
        probs,
        labels,
        policy,
        base_margins=base_margins,
        datasets=datasets,
    )
    metric = metrics_from_predictions(y_true, probs, pred, labels)
    if (
        metric["macro_f1"] < macro_floor
        or metric["accuracy"] < accuracy_floor
        or metric["ece"] > ece_ceiling
        or float(metric.get("tie_recall") or 0.0) <= baseline_tie
    ):
        return None
    objective = (
        metric["macro_f1"]
        + 0.25 * metric["accuracy"]
        + 0.20 * float(metric.get("tie_recall") or 0.0)
        - 0.05 * metric["ece"]
    )
    return {**policy, "objective": round(float(objective), 6), "metrics": metric}


def select_dataset_overlay_tie_policy(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    datasets: Sequence[str],
    *,
    base_policy: Dict[str, Any],
    base_margins: Optional[np.ndarray],
    macro_floor: float,
    accuracy_floor: float,
    ece_ceiling: float,
    baseline_tie: float,
) -> Optional[Dict[str, Any]]:
    if not base_policy.get("enabled"):
        return None
    dataset_values = np.array(list(datasets), dtype=object)
    tie_index = labels.index("Tie")
    best: Optional[Dict[str, Any]] = None
    base_dataset_policies = dict(base_policy.get("dataset_policies") or {})
    base_margin_candidates: Sequence[Optional[float]]
    base_margin_candidates = (None, 0.25, 0.50, 1.00) if base_margins is not None else (None,)
    for dataset in sorted(set(datasets)):
        indices = np.where(dataset_values == dataset)[0]
        if len(indices) < 40 or int((y_true[indices] == tie_index).sum()) < 8:
            continue
        for min_tie_probability in (0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
            for max_ab_margin in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
                for max_base_margin in base_margin_candidates:
                    dataset_policies = {
                        **base_dataset_policies,
                        dataset: {
                            "min_tie_probability": min_tie_probability,
                            "max_ab_margin": max_ab_margin,
                            "max_base_margin": max_base_margin,
                        },
                    }
                    policy = {
                        "enabled": True,
                        "min_tie_probability": base_policy.get("min_tie_probability"),
                        "max_ab_margin": base_policy.get("max_ab_margin"),
                        "max_base_margin": base_policy.get("max_base_margin"),
                        "dataset_policies": dataset_policies,
                    }
                    pred = apply_pairwise_tie_policy(
                        probs,
                        labels,
                        policy,
                        base_margins=base_margins,
                        datasets=datasets,
                    )
                    metric = metrics_from_predictions(y_true, probs, pred, labels)
                    if float(metric.get("tie_recall") or 0.0) <= baseline_tie:
                        continue
                    if metric["macro_f1"] < macro_floor:
                        continue
                    if metric["accuracy"] < accuracy_floor:
                        continue
                    if metric["ece"] > ece_ceiling:
                        continue
                    objective = (
                        metric["macro_f1"]
                        + 0.20 * metric["accuracy"]
                        + 0.30 * float(metric.get("tie_recall") or 0.0)
                        - 0.05 * metric["ece"]
                    )
                    row = {
                        **policy,
                        "objective": round(float(objective), 6),
                        "metrics": metric,
                        "overlay_dataset": dataset,
                    }
                    if best is None or (
                        float(metric.get("tie_recall") or 0.0),
                        metric["macro_f1"],
                        metric["accuracy"],
                        -metric["ece"],
                        objective,
                    ) > (
                        float(best["metrics"].get("tie_recall") or 0.0),
                        best["metrics"]["macro_f1"],
                        best["metrics"]["accuracy"],
                        -best["metrics"]["ece"],
                        float(best["objective"]),
                    ):
                        best = row
    return best


def feature_column(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    name: str,
) -> Optional[np.ndarray]:
    if name not in feature_names:
        return None
    return matrix[:, list(feature_names).index(name)]


def sample_datasets(samples: Sequence[Dict[str, Any]]) -> List[str]:
    return [normalize_text(sample.get("dataset")) for sample in samples]


def sample_weight_vector(
    samples: Sequence[Dict[str, Any]],
    *,
    class_weights: Optional[Dict[str, float]] = None,
    source_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    class_weights = class_weights or {}
    source_weights = source_weights or {}
    weights: List[float] = []
    for sample in samples:
        label = normalize_text(sample.get("human_label"))
        dataset = normalize_text(sample.get("dataset"))
        weights.append(float(class_weights.get(label, 1.0)) * float(source_weights.get(dataset, 1.0)))
    return np.array(weights, dtype=float)


def factuality_weight_candidates() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for unsupported_weight in (1.5, 2.0, 2.25, 2.5):
        for ragtruth_weight in (1.25, 1.5, 1.75):
            candidates.append(
                {
                    "class_weights": {"unsupported": unsupported_weight},
                    "source_weights": {"ragtruth": ragtruth_weight},
                }
            )
    return candidates


def apply_dataset_temperature_policy(
    probs: np.ndarray,
    datasets: Sequence[str],
    policy: Optional[Dict[str, Any]],
) -> np.ndarray:
    if not policy or not policy.get("enabled"):
        return probs
    temperatures = policy.get("temperatures", {})
    adjusted = probs.copy()
    for dataset, temperature in temperatures.items():
        indices = [i for i, name in enumerate(datasets) if name == dataset]
        if not indices:
            continue
        adjusted[indices] = temperature_scale(adjusted[indices], float(temperature))
    return adjusted


def select_dataset_temperature_policy(
    probs: np.ndarray,
    y_true: np.ndarray,
    labels: Sequence[str],
    datasets: Sequence[str],
    candidates: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if candidates is None:
        candidates = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
    baseline_pred = probs.argmax(axis=1)
    baseline_metrics = metrics_from_predictions(y_true, probs, baseline_pred, labels)
    temperatures: Dict[str, float] = {}
    rows: List[Dict[str, Any]] = []
    for dataset in sorted(set(datasets)):
        indices = [i for i, name in enumerate(datasets) if name == dataset]
        if len(indices) < 5:
            temperatures[dataset] = 1.0
            rows.append({"dataset": dataset, "count": len(indices), "temperature": 1.0, "skipped": True})
            continue
        subset_probs = probs[indices]
        subset_y = y_true[indices]
        best: Optional[Dict[str, Any]] = None
        for temperature in candidates:
            scaled = temperature_scale(subset_probs, float(temperature))
            nll = negative_log_likelihood(subset_y, scaled)
            ece = ece_score(subset_y, scaled)
            brier = brier_score(subset_y, scaled, len(labels))
            objective = nll + ece + 0.25 * brier
            row = {
                "dataset": dataset,
                "count": len(indices),
                "temperature": float(temperature),
                "nll": round(nll, 6),
                "ece": ece,
                "brier": brier,
                "objective": round(float(objective), 6),
                "skipped": False,
            }
            if best is None or objective < best["objective"]:
                best = row
        assert best is not None
        temperatures[dataset] = float(best["temperature"])
        rows.append(best)

    adjusted = apply_dataset_temperature_policy(
        probs,
        datasets,
        {"enabled": True, "temperatures": temperatures},
    )
    adjusted_pred = adjusted.argmax(axis=1)
    adjusted_metrics = metrics_from_predictions(y_true, adjusted, adjusted_pred, labels)
    enabled = adjusted_metrics["ece"] < baseline_metrics["ece"]
    return {
        "enabled": bool(enabled),
        "method": "dev_dataset_temperature_scaling",
        "selection_metric": "enable only when weighted dev ECE improves",
        "baseline_metrics": baseline_metrics,
        "selected_metrics": adjusted_metrics if enabled else baseline_metrics,
        "temperatures": temperatures if enabled else {},
        "candidates": rows,
    }


def metrics(y_true: np.ndarray, probs: np.ndarray, labels: Sequence[str]) -> Dict[str, Any]:
    pred = probs.argmax(axis=1)
    return metrics_from_predictions(y_true, probs, pred, labels)


def select_samples(samples: Sequence[Dict[str, Any]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    allowed = set(labels)
    return [sample for sample in samples if sample.get("human_label") in allowed]


def limit_samples(samples: Sequence[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    scoped = list(samples)
    if limit is None:
        return scoped
    return scoped[:limit]


def encode_labels(samples: Sequence[Dict[str, Any]], labels: Sequence[str]) -> np.ndarray:
    index = {label: i for i, label in enumerate(labels)}
    return np.array([index[str(sample["human_label"])] for sample in samples], dtype=int)


def encode_calibrated_rows(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = {label: i for i, label in enumerate(labels)}
    scoped = [
        row
        for row in rows
        if row.get("human_label") in index and row.get("predicted_label") in index
    ]
    y_true = np.array([index[str(row["human_label"])] for row in scoped], dtype=int)
    pred = np.array([index[str(row["predicted_label"])] for row in scoped], dtype=int)
    probs = np.array(
        [
            [float(row.get("label_probabilities", {}).get(label, 0.0)) for label in labels]
            for row in scoped
        ],
        dtype=float,
    )
    return y_true, pred, probs


def metrics_for_calibrated_rows(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> Dict[str, Any]:
    y_true, pred, probs = encode_calibrated_rows(rows, labels)
    if len(y_true) == 0:
        return {}
    return metrics_from_predictions(y_true, probs, pred, labels)


def by_dataset_table(rows: Sequence[Dict[str, Any]], labels: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in rows}):
        subset = [row for row in rows if str(row.get("dataset")) == dataset]
        metric = metrics_for_calibrated_rows(subset, labels)
        out.append(
            {
                "dataset": dataset,
                "count": len(subset),
                "accuracy": metric.get("accuracy"),
                "macro_f1": metric.get("macro_f1"),
                "ece": metric.get("ece"),
                "brier": metric.get("brier"),
                "review_rate": round(sum(1 for row in subset if row.get("review_flag")) / len(subset), 4)
                if subset
                else 0.0,
                "gold_distribution": metric.get("gold_distribution", {}),
                "pred_distribution": metric.get("pred_distribution", {}),
            }
        )
    return out


def ragtruth_error_table(rows: Sequence[Dict[str, Any]], *, limit_per_group: int = 25) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") != "ragtruth":
            continue
        gold = str(row.get("human_label"))
        pred = str(row.get("predicted_label"))
        confidence = float(row.get("confidence", 0.0))
        if gold != pred:
            groups[f"{gold}->{pred}"].append(row)
        elif confidence < 0.60:
            groups["low_confidence_correct"].append(row)
    out: List[Dict[str, Any]] = []
    for group, items in sorted(groups.items()):
        ordered = sorted(items, key=lambda row: float(row.get("confidence", 0.0)))
        for row in ordered[:limit_per_group]:
            out.append(
                {
                    "group": group,
                    "split": row.get("split"),
                    "id": row.get("id"),
                    "human_label": row.get("human_label"),
                    "predicted_label": row.get("predicted_label"),
                    "confidence": row.get("confidence"),
                    "risk_score": row.get("risk_score"),
                    "review_flag": row.get("review_flag"),
                    "review_reason": row.get("review_reason"),
                    "p_supported": row.get("label_probabilities", {}).get("supported"),
                    "p_unsupported": row.get("label_probabilities", {}).get("unsupported"),
                }
            )
    return out


def evidence_feature_ablation_table(
    samples: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sample_by_id = {str(sample.get("id")): sample for sample in samples}
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = sample_by_id.get(str(row.get("id")))
        if not sample:
            continue
        features = evidence_feature_dict(sample)
        bucket_names: List[str] = []
        if features.get("evidence_local_hallucination_risk_a", 0.0) >= 0.50:
            bucket_names.append("local_hallucination_risk")
        if features.get("evidence_low_support_sentence_ratio_a", 0.0) >= 0.50:
            bucket_names.append("low_support_sentence_ratio")
        if features.get("evidence_entity_gap_a", 0.0) >= 0.45:
            bucket_names.append("entity_gap")
        if features.get("evidence_entity_alias_gap_a", 0.0) >= 0.45:
            bucket_names.append("entity_alias_gap")
        if features.get("evidence_date_gap_a", 0.0) > 0.0 or features.get("evidence_numeric_gap_a", 0.0) > 0.0:
            bucket_names.append("numeric_or_date_gap")
        if features.get("evidence_negation_mismatch_a", 0.0) > 0.0:
            bucket_names.append("negation_mismatch")
        if features.get("evidence_comparative_mismatch_a", 0.0) > 0.0:
            bucket_names.append("comparative_mismatch")
        if not bucket_names:
            bucket_names.append("none")
        for bucket in bucket_names:
            buckets[bucket].append(row)
    out: List[Dict[str, Any]] = []
    for bucket, subset in sorted(buckets.items()):
        errors = [row for row in subset if row.get("human_label") != row.get("predicted_label")]
        out.append(
            {
                "evidence_subtype": bucket,
                "count": len(subset),
                "error_rate": round(len(errors) / len(subset), 4) if subset else 0.0,
                "review_rate": round(sum(1 for row in subset if row.get("review_flag")) / len(subset), 4)
                if subset
                else 0.0,
                "unsupported_rate": round(
                    sum(1 for row in subset if row.get("human_label") == "unsupported") / len(subset),
                    4,
                )
                if subset
                else 0.0,
            }
        )
    return out


def base_diagnostics_table(
    rows: Sequence[Dict[str, Any]],
    judge_outputs: JudgeOutputFeatures,
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    margins: Dict[Tuple[str, str, str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        base = judge_outputs.rows.get(str(row.get("id")))
        if not base:
            continue
        base_pred = str(base.get("pred_label"))
        calibrated_pred = str(row.get("predicted_label"))
        key = (
            str(row.get("split")),
            str(row.get("dataset")),
            str(row.get("human_label")),
            base_pred,
            calibrated_pred,
        )
        groups[key].append(row)
        margins[key].append(abs(float(base.get("score_a", 0.0)) - float(base.get("score_b", 0.0))))

    out: List[Dict[str, Any]] = []
    for key in sorted(groups):
        split, dataset, gold, base_pred, calibrated_pred = key
        subset = groups[key]
        margin_values = margins[key]
        out.append(
            {
                "split": split,
                "dataset": dataset,
                "gold_label": gold,
                "base_pred": base_pred,
                "calibrated_pred": calibrated_pred,
                "n": len(subset),
                "base_correct": base_pred == gold,
                "calibrated_correct": calibrated_pred == gold,
                "avg_base_margin": round(sum(margin_values) / len(margin_values), 4) if margin_values else 0.0,
                "review_rate": round(sum(1 for row in subset if row.get("review_flag")) / len(subset), 4),
            }
        )
    return out


def bias_group_from_sample(sample: Dict[str, Any]) -> str:
    meta = sample.get("metadata", {})
    candidates = [
        normalize_text(meta.get("bias_type")),
        normalize_text(meta.get("perturbation_applied")),
        normalize_text(meta.get("reasoning_difficulty")),
        normalize_text(meta.get("difficulty_type")),
    ]
    allowed = {"position", "length", "format", "rubric_sensitivity", "reasoning_difficulty"}
    for value in candidates:
        if value in allowed:
            return value
        if value in {"reasoning", "hard_reasoning", "complex_reasoning"}:
            return "reasoning_difficulty"
    return "none"


def bias_subgroup_calibration_table(
    samples: Sequence[Dict[str, Any]],
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sample_by_id = {str(sample.get("id")): sample for sample in samples}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    risks: Dict[str, List[float]] = defaultdict(list)
    required = {"position", "length", "format", "rubric_sensitivity", "reasoning_difficulty", "none"}
    for row in rows:
        sample = sample_by_id.get(str(row.get("id")))
        if not sample:
            continue
        group = bias_group_from_sample(sample)
        profile = build_bias_profile(sample)
        groups[group].append(row)
        risks[group].append(float(profile["bias"]["overall_bias_risk"]))

    out: List[Dict[str, Any]] = []
    for group in sorted(set(groups) | required):
        subset = groups.get(group, [])
        metric = metrics_for_calibrated_rows(subset, PAIRWISE_LABELS) if subset else {}
        risk_values = risks.get(group, [])
        out.append(
            {
                "bias_group": group,
                "n": len(subset),
                "accuracy": metric.get("accuracy", 0.0),
                "macro_f1": metric.get("macro_f1", 0.0),
                "ece": metric.get("ece", 0.0),
                "review_rate": round(sum(1 for row in subset if row.get("review_flag")) / len(subset), 4)
                if subset
                else 0.0,
                "avg_bias_risk": round(sum(risk_values) / len(risk_values), 4) if risk_values else 0.0,
            }
        )
    return out


def hyperparam_grid() -> List[HyperParams]:
    grid: List[HyperParams] = []
    for lr in (0.08, 0.04, 0.02, 0.01):
        for batch in (16, 32, 64):
            for l2 in (0.0, 1e-4, 1e-3, 5e-3):
                for epochs in (80, 140):
                    grid.append(HyperParams(lr, batch, l2, epochs))
    return grid


def evaluate_head_on_split(
    head: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    feature_fn,
    split_name: str,
) -> Dict[str, Any]:
    selected = select_samples(samples, labels)
    if not selected:
        return {
            "split": split_name,
            "count": 0,
            "metrics": {},
            "rows": [],
        }
    x_raw, _ = make_matrix(selected, feature_fn, head["feature_names"])
    x = apply_scaler(x_raw, head["scaler"])
    y = encode_labels(selected, labels)
    model = SoftmaxClassifier(len(head["feature_names"]), len(labels), seed=SEED)
    model.weights = np.array(head["model"]["weights"], dtype=float)
    model.bias = np.array(head["model"]["bias"], dtype=float)
    raw_probs = model.predict_proba(x)
    probs = temperature_scale(raw_probs, float(head["calibration"]["temperature"]))
    probs = apply_dataset_temperature_policy(
        probs,
        sample_datasets(selected),
        head.get("dataset_temperature_policy"),
    )
    base_margins = feature_column(x_raw, head["feature_names"], "base_margin")
    pred = apply_pairwise_tie_policy(
        probs,
        labels,
        head.get("tie_policy"),
        base_margins=base_margins,
        datasets=sample_datasets(selected),
    )
    if str(head.get("name")) == "factuality":
        pred = apply_factuality_threshold_policy(
            probs,
            labels,
            sample_datasets(selected),
            head.get("factuality_threshold_policy"),
        )
    rows = make_calibrated_rows(
        selected,
        labels,
        probs,
        review_threshold=float(head["review_policy"]["threshold"]),
        head_name=str(head["name"]),
        pred_indices=pred,
    )
    return {
        "split": split_name,
        "count": len(selected),
        "metrics": metrics_from_predictions(y, probs, pred, labels),
        "rows": rows,
    }


def head_raw_probs_on_split(
    head: Dict[str, Any],
    samples: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    feature_fn,
) -> Tuple[np.ndarray, np.ndarray]:
    selected = select_samples(samples, labels)
    if not selected:
        return np.zeros((0, len(labels)), dtype=float), np.zeros((0,), dtype=int)
    x_raw, _ = make_matrix(selected, feature_fn, head["feature_names"])
    x = apply_scaler(x_raw, head["scaler"])
    y = encode_labels(selected, labels)
    model = SoftmaxClassifier(len(head["feature_names"]), len(labels), seed=SEED)
    model.weights = np.array(head["model"]["weights"], dtype=float)
    model.bias = np.array(head["model"]["bias"], dtype=float)
    return model.predict_proba(x), y


def train_one_head(
    name: str,
    train_samples: Sequence[Dict[str, Any]],
    dev_samples: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    feature_fn,
    weight_candidates_override: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    x_train_raw, feature_names = make_matrix(train_samples, feature_fn)
    x_dev_raw, _ = make_matrix(dev_samples, feature_fn, feature_names)
    x_train, x_dev, scaler = standardize_train_dev(x_train_raw, x_dev_raw)
    y_train = encode_labels(train_samples, labels)
    y_dev = encode_labels(dev_samples, labels)

    trials: List[Dict[str, Any]] = []
    if weight_candidates_override is not None:
        weight_candidates = weight_candidates_override
    else:
        weight_candidates = factuality_weight_candidates() if name == "factuality" else [{"class_weights": {}, "source_weights": {}}]
    best: Optional[Tuple[float, SoftmaxClassifier, HyperParams, Dict[str, Any], List[float], Dict[str, Any]]] = None
    for params in hyperparam_grid():
        for weight_config in weight_candidates:
            model = SoftmaxClassifier(x_train.shape[1], len(labels), seed=SEED)
            train_weights = sample_weight_vector(
                train_samples,
                class_weights=weight_config.get("class_weights"),
                source_weights=weight_config.get("source_weights"),
            )
            losses = model.fit(x_train, y_train, params, seed=SEED, sample_weight=train_weights)
            probs = model.predict_proba(x_dev)
            dev_metrics = metrics(y_dev, probs, labels)
            objective = dev_metrics["macro_f1"] + 0.25 * dev_metrics["accuracy"] - 0.05 * dev_metrics["ece"]
            trial = {
                "learning_rate": params.learning_rate,
                "batch_size": params.batch_size,
                "l2": params.l2,
                "epochs": params.epochs,
                "class_weights": weight_config.get("class_weights", {}),
                "source_weights": weight_config.get("source_weights", {}),
                "objective": round(float(objective), 6),
                "dev_metrics": dev_metrics,
                "final_train_loss": losses[-1] if losses else None,
            }
            trials.append(trial)
            if best is None or objective > best[0]:
                best = (objective, model, params, dev_metrics, losses, weight_config)

    assert best is not None
    _objective, model, params, dev_metrics, losses, weight_config = best
    raw_dev_probs = model.predict_proba(x_dev)
    calibration = calibrate_temperature(raw_dev_probs, y_dev)
    calibrated_dev_probs = temperature_scale(raw_dev_probs, float(calibration["temperature"]))
    dataset_temperature_policy = (
        select_dataset_temperature_policy(
            calibrated_dev_probs,
            y_dev,
            labels,
            sample_datasets(dev_samples),
        )
        if name == "pairwise"
        else {
            "enabled": False,
            "method": "disabled_for_stable_factuality_head",
            "selection_metric": "not_applicable",
            "baseline_metrics": metrics_from_predictions(
                y_dev,
                calibrated_dev_probs,
                calibrated_dev_probs.argmax(axis=1),
                labels,
            ),
            "selected_metrics": metrics_from_predictions(
                y_dev,
                calibrated_dev_probs,
                calibrated_dev_probs.argmax(axis=1),
                labels,
            ),
            "temperatures": {},
            "candidates": [],
        }
    )
    calibrated_dev_probs = apply_dataset_temperature_policy(
        calibrated_dev_probs,
        sample_datasets(dev_samples),
        dataset_temperature_policy,
    )
    dev_base_margins = feature_column(x_dev_raw, feature_names, "base_margin")
    tie_policy = select_pairwise_tie_policy(
        calibrated_dev_probs,
        y_dev,
        labels,
        base_margins=dev_base_margins,
        datasets=sample_datasets(dev_samples),
    )
    calibrated_dev_pred = apply_pairwise_tie_policy(
        calibrated_dev_probs,
        labels,
        tie_policy,
        base_margins=dev_base_margins,
        datasets=sample_datasets(dev_samples),
    )
    factuality_threshold_policy = (
        select_factuality_threshold_policy(
            calibrated_dev_probs,
            y_dev,
            labels,
            sample_datasets(dev_samples),
        )
        if name == "factuality"
        else disabled_factuality_threshold_policy()
    )
    if name == "factuality":
        calibrated_dev_pred = apply_factuality_threshold_policy(
            calibrated_dev_probs,
            labels,
            sample_datasets(dev_samples),
            factuality_threshold_policy,
        )
    calibrated_dev_metrics = metrics_from_predictions(y_dev, calibrated_dev_probs, calibrated_dev_pred, labels)
    dev_confidence = confidence_for_predictions(calibrated_dev_probs, calibrated_dev_pred)
    dev_correct = calibrated_dev_pred == y_dev
    review_policy = select_review_threshold(risk_scores(dev_confidence), dev_correct)
    calibrated_dev_rows = make_calibrated_rows(
        dev_samples,
        labels,
        calibrated_dev_probs,
        review_threshold=float(review_policy["threshold"]),
        head_name=name,
        pred_indices=calibrated_dev_pred,
    )
    return {
        "name": name,
        "labels": list(labels),
        "feature_names": feature_names,
        "scaler": scaler,
        "best_hyperparameters": {
            "learning_rate": params.learning_rate,
            "batch_size": params.batch_size,
            "l2": params.l2,
            "epochs": params.epochs,
            "class_weights": weight_config.get("class_weights", {}),
            "source_weights": weight_config.get("source_weights", {}),
        },
        "dev_metrics": dev_metrics,
        "calibrated_dev_metrics": calibrated_dev_metrics,
        "calibration": calibration,
        "dataset_temperature_policy": dataset_temperature_policy,
        "tie_policy": tie_policy,
        "factuality_threshold_policy": factuality_threshold_policy,
        "review_policy": review_policy,
        "calibrated_dev_rows": calibrated_dev_rows,
        "loss_curve": losses,
        "model": model.to_dict(),
        "trials": sorted(trials, key=lambda row: row["objective"], reverse=True),
        "train_count": len(train_samples),
        "dev_count": len(dev_samples),
    }


def group_by_split(samples: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        out[str(sample.get("split", "NA"))].append(sample)
    return out


def validation_gate(report: Dict[str, Any]) -> Dict[str, Any]:
    pairwise = report["heads"]["pairwise"]["calibrated_dev_metrics"]
    factuality = report["heads"]["factuality"]["calibrated_dev_metrics"]
    gates = {
        "pairwise_accuracy_min_0_40": pairwise["accuracy"] >= 0.40,
        "pairwise_macro_f1_min_0_35": pairwise["macro_f1"] >= 0.35,
        "factuality_accuracy_min_0_70": factuality["accuracy"] >= 0.70,
        "factuality_macro_f1_min_0_695": factuality["macro_f1"] >= 0.695,
        "factuality_ece_max_0_20": factuality["ece"] <= 0.20,
    }
    return {
        "thresholds": {
            "pairwise_accuracy": 0.40,
            "pairwise_macro_f1": 0.35,
            "factuality_accuracy": 0.70,
            "factuality_macro_f1": 0.695,
            "factuality_ece": 0.20,
        },
        "checks": gates,
        "passed": all(gates.values()),
    }


def build_report_md(report: Dict[str, Any]) -> str:
    pair = report["heads"]["pairwise"]
    fact = report["heads"]["factuality"]
    gate = report["validation_gate"]
    test = report.get("test_evaluation", {})
    return "\n".join(
        [
            "# BEA-Judge Model Build Report",
            "",
            f"Created at: {report['created_at']}",
            f"Input dataset: `{report['input_dataset']}`",
            "",
            "## Model Structure",
            "",
            "- Base judge features: real Prometheus-family pairwise outputs, score gap, and predicted label indicators.",
            "- Bias-aware features: position, length, format, rubric-sensitivity, and synthetic perturbation flags.",
            "- Evidence features: context/reference overlap, sentence support, numeric/date/entity gaps, negation/comparative mismatch, and answer support proxies.",
            "- Calibration layer: task-specific softmax classifiers trained only on train split and selected on dev split.",
            "",
            "## Hyperparameter Selection",
            "",
            "Grid searched learning rate, batch size, L2 penalty, and epochs. The objective was macro-F1 + 0.25*accuracy - 0.05*ECE on dev.",
            "",
            f"Pairwise best params: `{pair['best_hyperparameters']}`",
            f"Factuality best params: `{fact['best_hyperparameters']}`",
            f"Pairwise calibration: temperature={pair['calibration']['temperature']}, review threshold={pair['review_policy']['threshold']}",
            f"Pairwise dataset temperature policy: `{pair.get('dataset_temperature_policy', {})}`",
            f"Pairwise tie policy: `{pair.get('tie_policy', {})}`",
            f"Factuality calibration: temperature={fact['calibration']['temperature']}, review threshold={fact['review_policy']['threshold']}",
            f"Factuality dataset temperature policy: `{fact.get('dataset_temperature_policy', {})}`",
            f"Factuality threshold policy: `{fact.get('factuality_threshold_policy', {})}`",
            "",
            "## Dev Metrics",
            "",
            f"Pairwise: accuracy={pair['dev_metrics']['accuracy']}, macro_f1={pair['dev_metrics']['macro_f1']}, ece={pair['dev_metrics']['ece']}, brier={pair['dev_metrics']['brier']}",
            f"Factuality: accuracy={fact['dev_metrics']['accuracy']}, macro_f1={fact['dev_metrics']['macro_f1']}, ece={fact['dev_metrics']['ece']}, brier={fact['dev_metrics']['brier']}",
            f"Pairwise calibrated: accuracy={pair['calibrated_dev_metrics']['accuracy']}, macro_f1={pair['calibrated_dev_metrics']['macro_f1']}, ece={pair['calibrated_dev_metrics']['ece']}, brier={pair['calibrated_dev_metrics']['brier']}",
            f"Factuality calibrated: accuracy={fact['calibrated_dev_metrics']['accuracy']}, macro_f1={fact['calibrated_dev_metrics']['macro_f1']}, ece={fact['calibrated_dev_metrics']['ece']}, brier={fact['calibrated_dev_metrics']['brier']}",
            "",
            "## Test-Only Metrics",
            "",
            f"Pairwise test: `{test.get('pairwise', {}).get('metrics', {})}`",
            f"Factuality test: `{test.get('factuality', {}).get('metrics', {})}`",
            "",
            "## Validation Gate",
            "",
            f"Passed: `{gate['passed']}`",
            f"Checks: `{gate['checks']}`",
        ]
    )


def write_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    if path.suffix.lower() == ".csv":
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                        for key, value in row.items()
                    }
                )
        return
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(str(row.get(key, "")).replace("\n", " ") for key in fieldnames) + " |"
        for row in rows
    ]
    path.write_text("\n".join([header, divider, *body]) + "\n", encoding="utf-8")


def update_experiment_config(config_path: Path, run_dir: Path) -> None:
    if not config_path.exists():
        return
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    latest = payload.setdefault("latest_outputs", {})
    latest["validation_report"] = path_relative_to_root(MODEL_OUT / "latest_validation_report.json")
    latest["calibrated_results"] = path_relative_to_root(run_dir / "calibrated_results.json")
    latest["run_directory"] = path_relative_to_root(run_dir)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(report: Dict[str, Any], run_name: Optional[str] = None) -> Path:
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name.strip()) if run_name else ""
    run_dir = MODEL_OUT / safe_run_name if safe_run_name else MODEL_OUT / f"bea_judge_{stamp}"
    if run_dir.exists() and safe_run_name:
        run_dir = MODEL_OUT / f"{safe_run_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    calibrated_results: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "train": {
            name: split_report["rows"]
            for name, split_report in report.get("train_evaluation", {}).items()
        },
        "dev": {
            name: head["calibrated_dev_rows"]
            for name, head in report["heads"].items()
        },
        "test": {
            name: split_report["rows"]
            for name, split_report in report.get("test_evaluation", {}).items()
        },
    }
    model_heads = {
        name: {key: value for key, value in head.items() if key != "calibrated_dev_rows"}
        for name, head in report["heads"].items()
    }
    report_for_disk = {
        **report,
        "heads": model_heads,
        "calibrated_results": {
            split: {
                name: {
                    "split": split,
                    "row_count": len(rows),
                    "path": path_relative_to_root(run_dir / "calibrated_results.json"),
                }
                for name, rows in split_rows.items()
            }
            for split, split_rows in calibrated_results.items()
        },
        "test_evaluation": {
            name: {key: value for key, value in split_report.items() if key != "rows"}
            for name, split_report in report.get("test_evaluation", {}).items()
        },
        "train_evaluation": {
            name: {key: value for key, value in split_report.items() if key != "rows"}
            for name, split_report in report.get("train_evaluation", {}).items()
        },
    }
    (run_dir / "model.json").write_text(json.dumps(model_heads, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "tuning_results.json").write_text(json.dumps(report["tuning"], ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "calibrated_results.json").write_text(
        json.dumps(calibrated_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "validation_report.json").write_text(
        json.dumps(report_for_disk, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "build_report.md").write_text(build_report_md(report), encoding="utf-8")
    diagnostics = report.get("diagnostics", {})
    for name, rows in diagnostics.items():
        if isinstance(rows, list):
            write_table(run_dir / f"{name}.json", rows)
            write_table(run_dir / f"{name}.md", rows)
    (MODEL_OUT / "latest_validation_report.json").write_text(
        json.dumps(report_for_disk, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    update_experiment_config(ROOT / "configs" / "experiment.json", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BEA-Judge lightweight model.")
    parser.add_argument("--input", type=str, default=str(PROCESSED / "bea_judge_core_2400.json"))
    parser.add_argument(
        "--judge-output",
        type=str,
        default=None,
        help="Path to base_scores.json generated by a real Prometheus-family judge run.",
    )
    parser.add_argument(
        "--base-scores",
        type=str,
        default=None,
        help="Alias for --judge-output, used by QLoRA experiment runbooks.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional model_outputs run directory name. Existing names receive a timestamp suffix.",
    )
    parser.add_argument(
        "--allow-heuristic-base",
        action="store_true",
        help="Allow heuristic_fallback rows for local prototype training; reports are not final model evidence.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional per-split cap for bounded smoke runs; applied after label filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.input)
    samples = read_dataset(dataset_path)
    judge_output_path = args.base_scores or args.judge_output
    if not judge_output_path:
        raise ValueError("one of --judge-output or --base-scores is required")
    judge_outputs = load_judge_output_features(Path(judge_output_path), allow_heuristic=args.allow_heuristic_base)
    if not judge_outputs.rows:
        raise ValueError(f"no valid real Prometheus-family judge rows found in {judge_output_path}")
    splits = group_by_split(samples)
    train = splits["train"]
    dev = splits["dev"]
    test = splits["test"]

    pair_train = limit_samples(select_samples(train, PAIRWISE_LABELS), args.sample_limit)
    pair_dev = limit_samples(select_samples(dev, PAIRWISE_LABELS), args.sample_limit)
    pair_test = limit_samples(select_samples(test, PAIRWISE_LABELS), args.sample_limit)
    fact_train = limit_samples(select_samples(train, FACTUALITY_LABELS), args.sample_limit)
    fact_dev = limit_samples(select_samples(dev, FACTUALITY_LABELS), args.sample_limit)
    fact_test = limit_samples(select_samples(test, FACTUALITY_LABELS), args.sample_limit)
    coverage_report = validate_judge_output_coverage(pair_train + pair_dev + pair_test, judge_outputs)
    pairwise_features = lambda sample: pairwise_feature_dict(sample, judge_outputs)
    factuality_active_labels = [
        label
        for label in FACTUALITY_LABELS
        if any(sample.get("human_label") == label for sample in fact_train + fact_dev)
    ]

    pairwise_head = train_one_head("pairwise", pair_train, pair_dev, PAIRWISE_LABELS, pairwise_features)
    factuality_head = train_one_head(
        "factuality",
        fact_train,
        fact_dev,
        factuality_active_labels,
        factuality_feature_dict,
    )
    pairwise_test = evaluate_head_on_split(
        pairwise_head,
        pair_test,
        PAIRWISE_LABELS,
        pairwise_features,
        "test",
    )
    pairwise_train_eval = evaluate_head_on_split(
        pairwise_head,
        pair_train,
        PAIRWISE_LABELS,
        pairwise_features,
        "train",
    )
    factuality_test = evaluate_head_on_split(
        factuality_head,
        fact_test,
        factuality_active_labels,
        factuality_feature_dict,
        "test",
    )
    factuality_train_eval = evaluate_head_on_split(
        factuality_head,
        fact_train,
        factuality_active_labels,
        factuality_feature_dict,
        "train",
    )
    pairwise_dev_rows = pairwise_head["calibrated_dev_rows"]
    pairwise_test_rows = pairwise_test["rows"]
    factuality_dev_rows = factuality_head["calibrated_dev_rows"]
    factuality_test_rows = factuality_test["rows"]
    diagnostics = {
        "base_diagnostics_table": base_diagnostics_table(
            pairwise_dev_rows + pairwise_test_rows,
            judge_outputs,
        ),
        "bias_subgroup_calibration_table": bias_subgroup_calibration_table(
            pair_dev + pair_test,
            pairwise_dev_rows + pairwise_test_rows,
        ),
        "factuality_by_dataset_table": by_dataset_table(
            factuality_dev_rows + factuality_test_rows,
            factuality_active_labels,
        ),
        "ragtruth_error_table": ragtruth_error_table(factuality_dev_rows + factuality_test_rows),
        "evidence_feature_ablation_table": evidence_feature_ablation_table(
            fact_dev + fact_test,
            factuality_dev_rows + factuality_test_rows,
        ),
    }

    report: Dict[str, Any] = {
        "created_at": utc_now(),
        "input_dataset": path_relative_to_root(dataset_path),
        "data_counts": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
            "pairwise_train": len(pair_train),
            "pairwise_dev": len(pair_dev),
            "pairwise_test": len(pair_test),
            "factuality_train": len(fact_train),
            "factuality_dev": len(fact_dev),
            "factuality_test": len(fact_test),
            "sample_limit_per_split": args.sample_limit,
        },
        "heads": {
            "pairwise": {
                key: value
                for key, value in pairwise_head.items()
                if key not in {"trials"}
            },
            "factuality": {
                key: value
                for key, value in factuality_head.items()
                if key not in {"trials"}
            },
        },
        "tuning": {
            "pairwise_top10": pairwise_head["trials"][:10],
            "factuality_top10": factuality_head["trials"][:10],
            "grid_size_per_head": len(hyperparam_grid()),
            "selection_objective": "macro_f1 + 0.25*accuracy - 0.05*ECE",
            "post_selection_calibration": {
                "method": "temperature scaling selected on dev nll + ece + 0.25*brier",
                "review_threshold": "lowest-confidence dev threshold selected to capture at least 80% of dev errors when possible",
            },
        },
        "test_evaluation": {
            "pairwise": pairwise_test,
            "factuality": factuality_test,
        },
        "train_evaluation": {
            "pairwise": pairwise_train_eval,
            "factuality": factuality_train_eval,
        },
        "diagnostics": diagnostics,
        "backbone": {
            "base_judge": "prometheus_family_real_outputs"
            if not args.allow_heuristic_base
            else "heuristic_fallback_local_prototype",
            "judge_output_path": path_relative_to_root(Path(judge_output_path)),
            "source_counts": judge_outputs.source_counts,
            "valid_pairwise_rows": len(judge_outputs.rows),
            "coverage": coverage_report,
            "final_evidence_ready": not args.allow_heuristic_base,
        },
    }
    report["validation_gate"] = validation_gate(report)
    run_dir = write_outputs(report, run_name=args.run_name)

    from calibration_methods import SUPPORTED_METHODS, run_calibration_comparison

    raw_dev_probs, y_dev_pair = head_raw_probs_on_split(
        pairwise_head, pair_dev, PAIRWISE_LABELS, pairwise_features
    )
    raw_test_probs, y_test_pair = head_raw_probs_on_split(
        pairwise_head, pair_test, PAIRWISE_LABELS, pairwise_features
    )
    if raw_dev_probs.shape[0] > 0 and raw_test_probs.shape[0] > 0:
        calibration_summary = run_calibration_comparison(
            p_dev=raw_dev_probs,
            y_dev=y_dev_pair,
            p_test=raw_test_probs,
            y_test=y_test_pair,
            methods=SUPPORTED_METHODS,
            out_dir=run_dir / "calibration_comparison",
            head="pairwise",
        )
        print(json.dumps(
            {
                method: result["metrics_test"]
                for method, result in calibration_summary["results"].items()
            },
            ensure_ascii=False,
            indent=2,
        ))
    print(json.dumps(report["data_counts"], ensure_ascii=False, indent=2))
    print(json.dumps(report["heads"]["pairwise"]["dev_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["heads"]["pairwise"]["calibrated_dev_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["heads"]["factuality"]["dev_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["heads"]["factuality"]["calibrated_dev_metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["test_evaluation"]["pairwise"]["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["test_evaluation"]["factuality"]["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(report["validation_gate"], ensure_ascii=False, indent=2))
    print(run_dir)


if __name__ == "__main__":
    main()
