"""
Canonical cleaner for BEA-Judge split datasets.

Transforms the current flat BEA-Judge split schema into the nested canonical
schema requested for downstream training, validation, and audit workflows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
CANONICAL_SCHEMA_VERSION = "2.0"
SINGLE_ANSWER_MARKER = "[SINGLE_ANSWER_FACTUALITY_TASK]"

TARGET_TOP_LEVEL_FIELDS = ["id", "source", "task", "input", "answers", "label", "quality", "metadata"]
SOURCE_DATASETS = {
    "mt_bench",
    "pandalm",
    "judgebench",
    "synthetic_perturbed",
    "ares_nq",
    "wikieval",
    "zh_professional",
    "helpsteer2",
    "ragtruth",
    "offsetbias",
    "oasst1",
    "rewardbench_external_eval",
}
TASK_TYPES = {"open_qa", "pairwise_bias", "factuality_rag"}
TASK_FORMS = {"pairwise", "single_answer"}
LANGUAGES = {"en", "zh"}
SPLITS = {"train", "dev", "test"}
LABEL_TYPES = {"pairwise_preference", "single_answer_factuality", "pairwise_factuality"}
LABEL_VALUES = {"A>B", "B>A", "Tie", "supported", "unsupported", "ambiguous"}
PAIRWISE_LABEL_VALUES = {"A>B": 1.0, "B>A": -1.0, "Tie": 0.0}
FACTUALITY_LABEL_VALUES = {"supported": 1.0, "ambiguous": 0.5, "unsupported": 0.0}

DATASET_NORMALIZATION = {
    "wikieval_grounded_vs_poor": "wikieval",
    "wikieval_grounded_vs_ungrounded": "wikieval",
    "zh_professional_open_qa": "zh_professional",
    "zh_professional_bias": "zh_professional",
    "zh_professional_factuality": "zh_professional",
}
SOURCE_LICENSES = {
    "mt_bench_human": "CC-BY-4.0",
    "pandalm_test": "Apache-2.0",
    "judgebench_claude": "MIT",
    "wikieval": None,
    "ares_nq_labeled": "Apache-2.0",
    "self_built_chinese_annotation": "internal/self-built",
    "helpsteer2": "CC-BY-4.0",
    "ragtruth": "MIT",
    "offsetbias": "BSD-3-Clause",
    "oasst1": "Apache-2.0",
    "rewardbench_external_eval": "mixed-subset-license",
}
SOURCE_DATASET_KEYS = {
    "mt_bench": "mt_bench_human",
    "pandalm": "pandalm_test",
    "judgebench": "judgebench_claude",
    "synthetic_perturbed": "judgebench_claude",
    "wikieval": "wikieval",
    "ares_nq": "ares_nq_labeled",
    "zh_professional": "self_built_chinese_annotation",
    "helpsteer2": "helpsteer2",
    "ragtruth": "ragtruth",
    "offsetbias": "offsetbias",
    "oasst1": "oasst1",
    "rewardbench_external_eval": "rewardbench_external_eval",
}


@dataclass
class TransformResult:
    record: Dict[str, Any]
    repairs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FileResult:
    input_path: Path
    output_path: Path
    split: str
    family: str
    input_count: int
    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    change_log: List[Dict[str, Any]] = field(default_factory=list)
    warning_count: int = 0
    repaired_count: int = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def as_optional_text(value: Any, field_path: str, repairs: List[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if not text.strip():
        repairs.append(f"optional_field_normalized_to_null:{field_path}")
        return None
    return text


def as_required_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            value = mapping[key]
            if not (isinstance(value, str) and not value.strip()):
                return value
    return None


def normalize_dataset_name(dataset: Any, repairs: List[str]) -> str:
    raw = str(dataset or "")
    normalized = DATASET_NORMALIZATION.get(raw, raw)
    if normalized != raw:
        repairs.append(f"dataset_normalized:{raw}->{normalized}")
    return normalized


def source_url_for(sample: Dict[str, Any], sources: List[str]) -> Optional[str]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    candidate = metadata.get("source_url")
    if isinstance(candidate, str) and candidate.startswith(("http://", "https://")):
        return candidate
    if len(sources) == 1 and isinstance(sources[0], str) and sources[0].startswith(("http://", "https://")):
        return sources[0]
    return None


def source_record_id_for(sample: Dict[str, Any]) -> Optional[str]:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    value = first_present(
        metadata,
        (
            "source_record_id",
            "original_idx",
            "original_index",
            "original_id",
            "pair_id",
            "row_id",
            "question_id",
        ),
    )
    return None if value is None else str(value)


def infer_task_form(sample: Dict[str, Any], answer_b: Any, repairs: List[str]) -> str:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    form = metadata.get("factuality_task_form")
    if form:
        return str(form)
    if sample.get("task_type") == "factuality_rag" and answer_b == SINGLE_ANSWER_MARKER:
        repairs.append("task_form_inferred:single_answer")
        return "single_answer"
    return "pairwise"


def infer_label_type(sample: Dict[str, Any]) -> Optional[str]:
    human_score = sample.get("human_score") if isinstance(sample.get("human_score"), dict) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    value = human_score.get("scoring_system") or metadata.get("scoring_system")
    return None if value is None else str(value)


def infer_label_score(sample: Dict[str, Any], label_type: Optional[str], label_value: Any) -> Optional[float]:
    human_score = sample.get("human_score") if isinstance(sample.get("human_score"), dict) else {}
    candidates: Tuple[str, ...]
    if label_type == "pairwise_preference":
        candidates = ("pairwise_preference",)
    elif label_type == "single_answer_factuality":
        candidates = ("factuality_label_score", "factuality_score_0_1")
    elif label_type == "pairwise_factuality":
        candidates = ("factuality_label_score", "pairwise_preference", "factuality_score_0_1")
    else:
        candidates = ()

    for key in candidates:
        value = human_score.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

    if label_type == "pairwise_preference" and label_value in PAIRWISE_LABEL_VALUES:
        return PAIRWISE_LABEL_VALUES[str(label_value)]
    if label_type == "single_answer_factuality" and label_value in FACTUALITY_LABEL_VALUES:
        return FACTUALITY_LABEL_VALUES[str(label_value)]
    return None


def build_quality(
    *,
    task_type: str,
    task_form: str,
    context: Optional[str],
    reference: Optional[str],
    answer_b: Optional[str],
) -> Dict[str, Any]:
    context_required = task_type == "factuality_rag"
    reference_required = False
    answer_b_required = task_form == "pairwise"
    missing_reason: Dict[str, str] = {}

    if context is None:
        missing_reason["context"] = "missing_required_field" if context_required else "not_required_for_task"
    if reference is None:
        missing_reason["reference"] = "not_required_for_task"
    if answer_b is None:
        missing_reason["answer_b"] = "missing_required_field" if answer_b_required else "single_answer_task"

    return {
        "context_required": context_required,
        "reference_required": reference_required,
        "answer_b_required": answer_b_required,
        "missing_reason": missing_reason,
    }


def transform_sample(sample: Dict[str, Any], containing_split: str, sources: List[str]) -> TransformResult:
    repairs: List[str] = []
    warnings: List[str] = []
    metadata = copy.deepcopy(sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {})
    dataset = normalize_dataset_name(sample.get("dataset"), repairs)
    source_url = source_url_for(sample, sources)
    source_record_id = source_record_id_for(sample)
    if source_url is None:
        warnings.append("source_url_null")
    if source_record_id is None:
        warnings.append("source_record_id_null")

    context = as_optional_text(sample.get("context"), "input.context", repairs)
    reference = as_optional_text(sample.get("reference"), "input.reference", repairs)
    raw_answer_b = sample.get("answer_b")
    task_form = infer_task_form(sample, raw_answer_b, repairs)
    if task_form == "single_answer" and raw_answer_b == SINGLE_ANSWER_MARKER:
        answer_b = None
        repairs.append("single_answer_marker_converted_to_null")
    else:
        answer_b = as_optional_text(raw_answer_b, "answers.b", repairs)

    task_type = str(sample.get("task_type") or "")
    label_type = infer_label_type(sample)
    label_value = sample.get("human_label")
    label_score = infer_label_score(sample, label_type, label_value)
    if label_score is None and label_value is not None:
        warnings.append("label_score_null")

    original_split = sample.get("split")
    if original_split != containing_split:
        repairs.append(f"split_corrected:{original_split}->{containing_split}")

    original_missing_values: Dict[str, str] = {}
    for source_field, target_field in (
        ("context", "input.context"),
        ("reference", "input.reference"),
        ("answer_b", "answers.b"),
    ):
        value = sample.get(source_field)
        if value is None:
            original_missing_values[target_field] = "null"
        elif isinstance(value, str) and not value.strip():
            original_missing_values[target_field] = "empty_string"

    if original_missing_values:
        metadata["original_missing_values"] = original_missing_values
    warnings.extend(repair for repair in repairs if repair.startswith("optional_field_normalized_to_null:"))
    metadata["canonical_transform"] = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "source_schema": "BEA-Judge flat split schema",
        "original_dataset": sample.get("dataset"),
        "original_split": original_split,
    }

    record = {
        "id": as_required_text(sample.get("id")),
        "source": {
            "dataset": dataset,
            "source_url": source_url,
            "source_record_id": source_record_id,
        },
        "task": {
            "type": task_type,
            "form": task_form,
            "language": str(sample.get("language") or ""),
            "split": containing_split,
        },
        "input": {
            "prompt": as_required_text(sample.get("prompt")),
            "context": context,
            "reference": reference,
        },
        "answers": {
            "a": as_required_text(sample.get("answer_a")),
            "b": answer_b,
        },
        "label": {
            "type": label_type,
            "value": label_value,
            "score": label_score,
        },
        "quality": build_quality(
            task_type=task_type,
            task_form=task_form,
            context=context,
            reference=reference,
            answer_b=answer_b,
        ),
        "metadata": metadata,
    }
    return TransformResult(record=record, repairs=repairs, warnings=warnings)


def validate_canonical_sample(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    if set(record) != set(TARGET_TOP_LEVEL_FIELDS):
        errors.append("top_level_fields_not_canonical")

    source = record.get("source") if isinstance(record.get("source"), dict) else {}
    task = record.get("task") if isinstance(record.get("task"), dict) else {}
    input_block = record.get("input") if isinstance(record.get("input"), dict) else {}
    answers = record.get("answers") if isinstance(record.get("answers"), dict) else {}
    label = record.get("label") if isinstance(record.get("label"), dict) else {}
    quality = record.get("quality") if isinstance(record.get("quality"), dict) else {}

    if is_blank(record.get("id")):
        errors.append("id_missing")
    if source.get("dataset") not in SOURCE_DATASETS:
        errors.append("source.dataset_invalid")
    if source.get("source_url") is None:
        warnings.append("source.source_url_null")
    if source.get("source_record_id") is None:
        warnings.append("source.source_record_id_null")
    if task.get("type") not in TASK_TYPES:
        errors.append("task.type_invalid")
    if task.get("form") not in TASK_FORMS:
        errors.append("task.form_invalid")
    if task.get("language") not in LANGUAGES:
        errors.append("task.language_invalid")
    if task.get("split") not in SPLITS:
        errors.append("task.split_invalid")
    if is_blank(input_block.get("prompt")):
        errors.append("input.prompt_missing")
    if is_blank(answers.get("a")):
        errors.append("answers.a_missing")
    if label.get("type") not in LABEL_TYPES:
        errors.append("label.type_invalid")
    if label.get("value") not in LABEL_VALUES:
        errors.append("label.value_invalid")
    if label.get("value") is not None and label.get("score") is None:
        warnings.append("label.score_null")

    if task.get("type") in {"open_qa", "pairwise_bias"} and label.get("type") != "pairwise_preference":
        errors.append("label.type_inconsistent_with_task")
    if task.get("type") == "factuality_rag" and label.get("type") not in {
        "single_answer_factuality",
        "pairwise_factuality",
    }:
        errors.append("label.type_inconsistent_with_task")
    if task.get("form") == "pairwise" and is_blank(answers.get("b")):
        errors.append("answers.b_missing_for_pairwise")
    if task.get("type") == "factuality_rag" and is_blank(input_block.get("context")):
        errors.append("input.context_missing_for_factuality")

    if not isinstance(quality, dict):
        errors.append("quality_missing")
    else:
        if quality.get("context_required") != (task.get("type") == "factuality_rag"):
            errors.append("quality.context_required_inconsistent")
        if quality.get("answer_b_required") != (task.get("form") == "pairwise"):
            errors.append("quality.answer_b_required_inconsistent")
        if not isinstance(quality.get("missing_reason"), dict):
            errors.append("quality.missing_reason_invalid")

    for path, value in (
        ("input.context", input_block.get("context")),
        ("input.reference", input_block.get("reference")),
        ("answers.b", answers.get("b")),
        ("source.source_url", source.get("source_url")),
        ("source.source_record_id", source.get("source_record_id")),
    ):
        if isinstance(value, str) and not value.strip():
            errors.append(f"{path}_empty_string")

    return errors, warnings


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, child_prefix))
        return out
    return {prefix: value}


def missing_kind(value: Any) -> Optional[str]:
    if value is None:
        return "null"
    if isinstance(value, str):
        if value == "":
            return "empty_string"
        if not value.strip():
            return "whitespace_only"
    return None


def field_missing_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    flattened = [flatten(record) for record in records]
    fields = sorted(set().union(*(row.keys() for row in flattened)) if flattened else [])
    row_count = len(records)
    report: Dict[str, Any] = {}
    for field_name in fields:
        counts = Counter()
        for row in flattened:
            if field_name not in row:
                counts["absent"] += 1
                continue
            kind = missing_kind(row[field_name])
            if kind:
                counts[kind] += 1
        missing_total = counts["absent"] + counts["null"] + counts["empty_string"] + counts["whitespace_only"]
        if missing_total:
            report[field_name] = {
                "absent": counts["absent"],
                "null": counts["null"],
                "empty_string": counts["empty_string"],
                "whitespace_only": counts["whitespace_only"],
                "missing_total": missing_total,
                "missing_ratio": round(missing_total / row_count, 6) if row_count else 0.0,
            }
    return report


def content_key(record: Dict[str, Any]) -> str:
    return sha256_text(
        "\u241f".join(
            [
                str(record.get("task", {}).get("type")),
                str(record.get("input", {}).get("prompt")),
                str(record.get("answers", {}).get("a")),
                str(record.get("answers", {}).get("b")),
            ]
        )
    )


def split_leakage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_content: Dict[str, set] = defaultdict(set)
    for record in records:
        by_content[content_key(record)].add(record["task"]["split"])
    leaked = [key for key, splits in by_content.items() if len(splits) > 1]
    return {"content_cross_split_leakage_count": len(leaked), "examples": leaked[:10]}


def clean_file(input_path: Path, output_path: Path, family: str, split: str) -> FileResult:
    payload = load_json(input_path)
    samples = payload.get("samples", [])
    sources = payload.get("dataset_info", {}).get("sources", [])
    result = FileResult(
        input_path=input_path,
        output_path=output_path,
        split=split,
        family=family,
        input_count=len(samples),
    )
    accepted: List[Dict[str, Any]] = []
    for sample in samples:
        transformed = transform_sample(sample, split, sources)
        errors, validation_warnings = validate_canonical_sample(transformed.record)
        warnings = sorted(set(transformed.warnings + validation_warnings))
        repairs = sorted(set(transformed.repairs))
        if errors:
            result.rejected.append(
                {
                    "id": sample.get("id"),
                    "file": str(input_path.relative_to(ROOT)),
                    "errors": errors,
                    "original_dataset": sample.get("dataset"),
                }
            )
            continue
        accepted.append(transformed.record)
        if repairs or warnings:
            result.change_log.append(
                {
                    "id": transformed.record["id"],
                    "file": str(input_path.relative_to(ROOT)),
                    "repairs": repairs,
                    "warnings": warnings,
                }
            )
        if repairs:
            result.repaired_count += 1
        if warnings:
            result.warning_count += 1

    wrapper = {
        "dataset_info": {
            "name": f"BEA-Judge canonical {family} {split} split",
            "schema": "BEA-Judge canonical nested schema",
            "schema_version": CANONICAL_SCHEMA_VERSION,
            "created_at": payload.get("dataset_info", {}).get("created_at"),
            "processed_at": utc_now(),
            "source_file": str(input_path.relative_to(ROOT)),
            "sample_count": len(accepted),
            "fields": TARGET_TOP_LEVEL_FIELDS,
        },
        "samples": accepted,
    }
    write_json(output_path, wrapper)
    result.accepted = accepted
    return result


def reject_duplicates(file_results: List[FileResult]) -> None:
    seen_ids: set = set()
    seen_content: set = set()
    for file_result in file_results:
        kept: List[Dict[str, Any]] = []
        for record in file_result.accepted:
            duplicate_errors = []
            if record["id"] in seen_ids:
                duplicate_errors.append("duplicate_id")
            key = content_key(record)
            if key in seen_content:
                duplicate_errors.append("duplicate_content")
            if duplicate_errors:
                file_result.rejected.append(
                    {
                        "id": record["id"],
                        "file": str(file_result.input_path.relative_to(ROOT)),
                        "errors": duplicate_errors,
                        "original_dataset": record["metadata"].get("canonical_transform", {}).get("original_dataset"),
                    }
                )
                continue
            seen_ids.add(record["id"])
            seen_content.add(key)
            kept.append(record)
        file_result.accepted = kept
        payload = load_json(file_result.output_path)
        payload["samples"] = kept
        payload["dataset_info"]["sample_count"] = len(kept)
        write_json(file_result.output_path, payload)


def source_completeness(manifest: Dict[str, Any]) -> Dict[str, Any]:
    rows = {}
    for key, item in manifest.get("sources", {}).items():
        rows[key] = {
            "license": item.get("license"),
            "license_status": "present" if item.get("license") else "missing",
            "acquired_at": item.get("acquired_at"),
            "acquisition_status": "present" if item.get("acquired_at") else "missing",
            "sha256": item.get("sha256"),
            "preprocessing_schema_version": item.get("preprocessing_schema_version"),
        }
    return rows


def update_manifest(input_manifest_path: Path, output_manifest_path: Path) -> Dict[str, Any]:
    manifest = load_json(input_manifest_path) if input_manifest_path.exists() else {"sources": {}}
    manifest = copy.deepcopy(manifest)
    manifest["updated_at"] = utc_now()
    manifest["canonical_schema_version"] = CANONICAL_SCHEMA_VERSION
    sources = manifest.setdefault("sources", {})

    for key, item in list(sources.items()):
        path_text = item.get("path")
        raw_path = ROOT / path_text if path_text else None
        if raw_path and raw_path.exists():
            item["acquired_at"] = datetime.fromtimestamp(raw_path.stat().st_mtime, timezone.utc).isoformat()
            item["sha256"] = item.get("sha256") or sha256_file(raw_path)
        else:
            item.setdefault("acquired_at", None)
        item["license"] = SOURCE_LICENSES.get(key)
        item["license_status"] = "present" if item["license"] else "missing"
        item["preprocessing_schema_version"] = CANONICAL_SCHEMA_VERSION

    sources.setdefault(
        "self_built_chinese_annotation",
        {
            "url": None,
            "path": "datasets\\processed\\chinese_professional_annotated_latest.json",
            "license": SOURCE_LICENSES["self_built_chinese_annotation"],
            "license_status": "present",
            "acquired_at": None,
            "sha256": None,
            "preprocessing_schema_version": CANONICAL_SCHEMA_VERSION,
        },
    )
    zh_item = sources["self_built_chinese_annotation"]
    zh_path = ROOT / zh_item["path"]
    if zh_path.exists():
        zh_item["acquired_at"] = datetime.fromtimestamp(zh_path.stat().st_mtime, timezone.utc).isoformat()
        zh_item["sha256"] = sha256_file(zh_path)

    write_json(output_manifest_path, manifest)
    return manifest


def build_markdown_report(report: Dict[str, Any]) -> str:
    lines = [
        "# Canonical Dataset Cleaning Validation Report",
        "",
        f"- Generated at: {report['created_at']}",
        f"- Canonical schema version: {CANONICAL_SCHEMA_VERSION}",
        f"- Input records: {report['summary']['input_records']}",
        f"- Accepted records: {report['summary']['accepted_records']}",
        f"- Repaired records: {report['summary']['repaired_records']}",
        f"- Warned records: {report['summary']['warned_records']}",
        f"- Rejected records: {report['summary']['rejected_records']}",
        f"- Schema compliance: {report['schema_compliance']}",
        "",
        "## Validation Gates",
        "",
    ]
    for key, value in report["validation_gates"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Source Metadata", ""])
    for key, value in report["source_completeness"].items():
        lines.append(
            f"- {key}: license={value['license'] or 'MISSING'}, "
            f"acquired_at={value['acquired_at'] or 'MISSING'}, sha256={value['sha256'] or 'MISSING'}"
        )
    lines.extend(["", "## Residual Risks", ""])
    if report["residual_risks"]:
        lines.extend(f"- {risk}" for risk in report["residual_risks"])
    else:
        lines.append("- No residual risks recorded.")
    return "\n".join(lines) + "\n"


def build_validation_report(file_results: List[FileResult], manifest: Dict[str, Any]) -> Dict[str, Any]:
    accepted = [record for file_result in file_results for record in file_result.accepted]
    rejected = [row for file_result in file_results for row in file_result.rejected]
    change_log = [row for file_result in file_results for row in file_result.change_log]
    duplicate_id_count = len(accepted) - len({record["id"] for record in accepted})
    duplicate_content_count = len(accepted) - len({content_key(record) for record in accepted})
    leakage = split_leakage(accepted)
    enum_errors = Counter()
    required_errors = Counter()
    empty_optional_strings = 0
    for record in accepted:
        errors, _warnings = validate_canonical_sample(record)
        for error in errors:
            if error.endswith("_invalid"):
                enum_errors[error] += 1
            if "missing" in error:
                required_errors[error] += 1
            if error.endswith("_empty_string"):
                empty_optional_strings += 1

    source_meta = source_completeness(manifest)
    residual_risks = []
    for key, item in source_meta.items():
        if item["license_status"] == "missing":
            residual_risks.append(f"Source `{key}` is missing explicit license metadata.")
        if item["acquisition_status"] == "missing":
            residual_risks.append(f"Source `{key}` is missing acquisition timestamp.")
        if not item.get("sha256"):
            residual_risks.append(f"Source `{key}` is missing SHA256 provenance hash.")
    if any("source_record_id_null" in row["warnings"] for row in change_log):
        residual_risks.append("Some records have null source.source_record_id and require source-level traceability review.")
    if any("source_url_null" in row["warnings"] for row in change_log):
        residual_risks.append("Some records have null source.source_url, mainly self-built or constructed sources.")

    report = {
        "created_at": utc_now(),
        "summary": {
            "input_records": sum(file_result.input_count for file_result in file_results),
            "accepted_records": len(accepted),
            "repaired_records": sum(file_result.repaired_count for file_result in file_results),
            "warned_records": sum(file_result.warning_count for file_result in file_results),
            "rejected_records": len(rejected),
        },
        "files": [
            {
                "input": str(file_result.input_path.relative_to(ROOT)),
                "output": str(file_result.output_path.relative_to(ROOT)),
                "input_count": file_result.input_count,
                "accepted_count": len(file_result.accepted),
                "rejected_count": len(file_result.rejected),
                "repaired_count": file_result.repaired_count,
                "warned_count": file_result.warning_count,
            }
            for file_result in file_results
        ],
        "field_missing_before": {},
        "field_missing_after": field_missing_report(accepted),
        "enum_violation_summary": dict(enum_errors),
        "required_field_error_summary": dict(required_errors),
        "duplicate_summary": {
            "duplicate_id_count": duplicate_id_count,
            "duplicate_content_count": duplicate_content_count,
        },
        "split_leakage_summary": leakage,
        "validation_gates": {
            "canonical_schema_errors": sum(1 for record in accepted if validate_canonical_sample(record)[0]),
            "empty_optional_string_errors": empty_optional_strings,
            "duplicate_id_count": duplicate_id_count,
            "duplicate_content_count": duplicate_content_count,
            "cross_split_content_leakage_count": leakage["content_cross_split_leakage_count"],
            "invalid_enum_errors": sum(enum_errors.values()),
            "required_field_errors": sum(required_errors.values()),
        },
        "source_completeness": source_meta,
        "residual_risks": residual_risks,
        "rejected_record_examples": rejected[:20],
    }
    before_records: List[Dict[str, Any]] = []
    for file_result in file_results:
        payload = load_json(file_result.input_path)
        before_records.extend(payload.get("samples", []))
    report["field_missing_before"] = field_missing_report(before_records)
    report["schema_compliance"] = (
        "pass"
        if report["validation_gates"]["canonical_schema_errors"] == 0
        and report["validation_gates"]["empty_optional_string_errors"] == 0
        and duplicate_id_count == 0
        and duplicate_content_count == 0
        and leakage["content_cross_split_leakage_count"] == 0
        and sum(enum_errors.values()) == 0
        and sum(required_errors.values()) == 0
        and not rejected
        else "fail"
    )
    return report


def default_jobs(datasets_root: Path) -> List[Tuple[str, str, Path, Path]]:
    return [
        ("core", "train", datasets_root / "splits" / "train.json", datasets_root / "cleaned" / "train.json"),
        ("core", "dev", datasets_root / "splits" / "dev.json", datasets_root / "cleaned" / "dev.json"),
        ("core", "test", datasets_root / "splits" / "test.json", datasets_root / "cleaned" / "test.json"),
        ("zh", "train", datasets_root / "splits_zh" / "train.json", datasets_root / "cleaned_zh" / "train.json"),
        ("zh", "dev", datasets_root / "splits_zh" / "dev.json", datasets_root / "cleaned_zh" / "dev.json"),
        ("zh", "test", datasets_root / "splits_zh" / "test.json", datasets_root / "cleaned_zh" / "test.json"),
    ]


def run_cleaning(datasets_root: Path = DATASETS) -> Dict[str, Any]:
    file_results = [
        clean_file(input_path, output_path, family, split)
        for family, split, input_path, output_path in default_jobs(datasets_root)
    ]
    reject_duplicates(file_results)
    manifest = update_manifest(
        datasets_root / "data_manifest.json",
        datasets_root / "updated_data_manifest.json",
    )
    report = build_validation_report(file_results, manifest)
    write_json(datasets_root / "cleaning_validation_report.json", report)
    write_json(
        datasets_root / "rejected_records.json",
        {
            "created_at": utc_now(),
            "records": [row for file_result in file_results for row in file_result.rejected],
        },
    )
    write_json(
        datasets_root / "cleaning_change_log.json",
        {
            "created_at": utc_now(),
            "records": [row for file_result in file_results for row in file_result.change_log],
        },
    )
    (datasets_root / "cleaning_validation_report.md").write_text(build_markdown_report(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean BEA-Judge splits into canonical nested schema.")
    parser.add_argument("--datasets", type=Path, default=DATASETS, help="Dataset root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_cleaning(args.datasets)
    print(
        json.dumps(
            {
                "schema_compliance": report["schema_compliance"],
                "summary": report["summary"],
                "validation_gates": report["validation_gates"],
                "report": str((args.datasets / "cleaning_validation_report.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
