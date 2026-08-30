"""Adapters between BEA-Judge canonical and legacy flat sample schemas."""

from __future__ import annotations

import copy
from typing import Any, Dict, List


CANONICAL_TOP_LEVEL = {"id", "source", "task", "input", "answers", "label", "quality", "metadata"}


def is_canonical_sample(sample: Dict[str, Any]) -> bool:
    return CANONICAL_TOP_LEVEL.issubset(sample.keys()) and isinstance(sample.get("task"), dict)


def _label_score_fields(label_type: str, score: Any) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    if score is None:
        return fields
    if label_type == "pairwise_preference":
        fields["pairwise_preference"] = score
    elif label_type == "single_answer_factuality":
        fields["factuality_label_score"] = score
        fields["factuality_score_0_1"] = score
    elif label_type == "pairwise_factuality":
        fields["factuality_label_score"] = score
    return fields


def canonical_to_flat_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    source = sample.get("source") or {}
    task = sample.get("task") or {}
    input_block = sample.get("input") or {}
    answers = sample.get("answers") or {}
    label = sample.get("label") or {}
    quality = sample.get("quality") or {}
    metadata = copy.deepcopy(sample.get("metadata") or {})

    label_type = label.get("type")
    label_value = label.get("value")
    label_score = label.get("score")
    human_score = {
        "score_format": label_type,
        "scoring_system": label_type,
        "label": label_value,
        **_label_score_fields(str(label_type), label_score),
    }

    source_url = source.get("source_url")
    source_record_id = source.get("source_record_id")
    if source_url is not None:
        metadata.setdefault("source_url", source_url)
    if source_record_id is not None:
        metadata.setdefault("source_record_id", str(source_record_id))
    if task.get("form") is not None:
        metadata.setdefault("factuality_task_form", task.get("form"))
    metadata.setdefault("canonical_quality", quality)
    metadata.setdefault("canonical_source", source)

    return {
        "id": sample.get("id"),
        "dataset": source.get("dataset"),
        "task_type": task.get("type"),
        "prompt": input_block.get("prompt"),
        "context": input_block.get("context"),
        "answer_a": answers.get("a"),
        "answer_b": answers.get("b"),
        "reference": input_block.get("reference"),
        "human_score": human_score,
        "human_label": label_value,
        "language": task.get("language"),
        "split": task.get("split"),
        "metadata": metadata,
    }


def adapt_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    if is_canonical_sample(sample):
        return canonical_to_flat_sample(sample)
    return sample


def adapt_dataset_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        return payload
    return {**payload, "samples": [adapt_sample(sample) for sample in samples]}


def samples_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    return adapt_dataset_payload(payload)["samples"]
