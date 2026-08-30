"""
Compliance audit for BEA-Judge datasets.

Checks the concrete schema and governance risks raised during dataset review:
- factuality labels and single-answer task marking
- human_score/metadata scoring-system consistency
- core 1200 subset relation to core 2400
- repeated prompts and split leakage
- phone-like PII candidates
- very short answers
- Chinese factuality context completeness
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
PROCESSED = DATASETS / "processed"
PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b")


def text_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def blank_string_paths(value: Any, path: str = "$") -> List[str]:
    if isinstance(value, str):
        return [path] if not value.strip() else []
    if isinstance(value, dict):
        paths: List[str] = []
        for key, child in value.items():
            paths.extend(blank_string_paths(child, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        paths: List[str] = []
        for idx, child in enumerate(value):
            paths.extend(blank_string_paths(child, f"{path}[{idx}]"))
        return paths
    return []


def read_samples(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["samples"]


def task_prompt_leaks(samples: List[Dict[str, Any]]) -> int:
    by_prompt: Dict[tuple, set] = defaultdict(set)
    for sample in samples:
        by_prompt[(sample.get("task_type"), sample.get("prompt"))].add(sample.get("split"))
    return sum(1 for splits in by_prompt.values() if len(splits) > 1)


def content_duplicates(samples: List[Dict[str, Any]]) -> int:
    seen = set()
    duplicate_count = 0
    for sample in samples:
        key = (
            sample.get("task_type"),
            sample.get("prompt"),
            sample.get("answer_a"),
            sample.get("answer_b"),
        )
        if key in seen:
            duplicate_count += 1
        seen.add(key)
    return duplicate_count


def phone_candidates(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    fields = ("prompt", "context", "answer_a", "answer_b", "reference")
    for sample in samples:
        matches = []
        for field in fields:
            text = text_value(sample.get(field))
            for match in PHONE_RE.findall(text):
                matches.append({"field": field, "match": match})
        if matches:
            review = sample.get("metadata", {}).get("pii_review", {})
            retained_findings = [
                finding
                for finding in review.get("findings", [])
                if finding.get("action") == "numeric_false_positive_retained"
            ]
            if review.get("status") == "reviewed" and len(retained_findings) == len(matches):
                continue
            candidates.append(
                {
                    "id": sample.get("id"),
                    "task_type": sample.get("task_type"),
                    "dataset": sample.get("dataset"),
                    "matches": matches[:5],
                    "review_status": "requires_manual_review",
                }
            )
    return candidates


def audit_core() -> Dict[str, Any]:
    core2400_path = PROCESSED / "bea_judge_core_2400.json"
    core1200_path = PROCESSED / "bea_judge_core_1200.json"
    core = read_samples(core2400_path)
    subset = read_samples(core1200_path)

    factuality = [sample for sample in core if sample.get("task_type") == "factuality_rag"]
    open_qa = [sample for sample in core if sample.get("task_type") == "open_qa"]

    factuality_null_labels = [sample["id"] for sample in factuality if sample.get("human_label") is None]
    factuality_empty_answer_b = [sample["id"] for sample in factuality if not sample.get("answer_b")]
    single_answer_bad_marker = [
        sample["id"]
        for sample in factuality
        if sample.get("metadata", {}).get("factuality_task_form") == "single_answer"
        and sample.get("answer_b") != "[SINGLE_ANSWER_FACTUALITY_TASK]"
    ]

    score_format_missing = [
        sample["id"]
        for sample in core
        if not sample.get("metadata", {}).get("score_format")
        or not sample.get("metadata", {}).get("scoring_system")
        or not sample.get("human_score", {}).get("score_format")
        or not sample.get("human_score", {}).get("scoring_system")
    ]

    open_score_formats = Counter(sample.get("metadata", {}).get("score_format", "missing") for sample in open_qa)

    core_ids = {sample["id"] for sample in core}
    subset_ids = {sample["id"] for sample in subset}
    subset_extra_ids = sorted(subset_ids - core_ids)

    repeated_prompts = Counter(sample.get("prompt") for sample in open_qa)
    repeated_prompt_count = sum(1 for count in repeated_prompts.values() if count > 1)
    max_prompt_repeats = max(repeated_prompts.values()) if repeated_prompts else 0

    short_non_factuality = [
        {
            "id": sample["id"],
            "task_type": sample.get("task_type"),
            "dataset": sample.get("dataset"),
            "answer_a_chars": len(text_value(sample.get("answer_a"))),
        }
        for sample in core
        if sample.get("task_type") != "factuality_rag" and len(text_value(sample.get("answer_a"))) < 10
    ]

    phones = phone_candidates(core)

    return {
        "dataset": str(core2400_path.relative_to(ROOT)),
        "sample_count": len(core),
        "status": {
            "c1_factuality_null_human_label": "pass" if not factuality_null_labels else "fail",
            "c1_factuality_empty_answer_b": "pass" if not factuality_empty_answer_b else "fail",
            "c1_single_answer_marker": "pass" if not single_answer_bad_marker else "fail",
            "c2_score_format_declared": "pass" if not score_format_missing else "fail",
            "c3_core1200_subset_of_core2400": "pass" if not subset_extra_ids else "fail",
            "split_leakage": "pass" if task_prompt_leaks(core) == 0 else "fail",
            "content_duplicates": "pass" if content_duplicates(core) == 0 else "fail",
            "short_non_factuality_answers": "pass" if not short_non_factuality else "review",
            "phone_like_candidates": "pass" if not phones else "review",
        },
        "counts": {
            "factuality_total": len(factuality),
            "factuality_null_human_label": len(factuality_null_labels),
            "factuality_empty_answer_b": len(factuality_empty_answer_b),
            "single_answer_bad_marker": len(single_answer_bad_marker),
            "score_format_missing": len(score_format_missing),
            "open_qa_score_formats": dict(open_score_formats),
            "core1200_extra_ids_not_in_core2400": len(subset_extra_ids),
            "open_qa_repeated_prompt_count": repeated_prompt_count,
            "open_qa_max_prompt_repeats": max_prompt_repeats,
            "task_prompt_split_leakage": task_prompt_leaks(core),
            "content_duplicates": content_duplicates(core),
            "short_non_factuality_answer_count": len(short_non_factuality),
            "phone_like_candidate_count": len(phones),
        },
        "review_items": {
            "phone_like_candidates": phones,
            "short_non_factuality_answers": short_non_factuality,
            "core1200_extra_ids_not_in_core2400": subset_extra_ids[:20],
        },
    }


def audit_chinese() -> Dict[str, Any]:
    path = PROCESSED / "chinese_professional_annotated_latest.json"
    samples = read_samples(path)
    factuality = [sample for sample in samples if sample.get("task_type") == "factuality_rag"]
    factuality_empty_context = [sample["id"] for sample in factuality if not sample.get("context")]
    non_factuality_context_not_null = [
        sample["id"]
        for sample in samples
        if sample.get("task_type") != "factuality_rag" and sample.get("context") is not None
    ]
    score_format_missing = [
        sample["id"]
        for sample in samples
        if not sample.get("metadata", {}).get("score_format")
        or not sample.get("metadata", {}).get("scoring_system")
        or not sample.get("human_score", {}).get("score_format")
        or not sample.get("human_score", {}).get("scoring_system")
    ]
    blank_string_locations = []
    for sample in samples:
        blank_string_locations.extend({"id": sample.get("id"), "path": path} for path in blank_string_paths(sample))
    field_contract_missing = [
        sample["id"]
        for sample in samples
        if not sample.get("metadata", {}).get("field_contract")
        or sample.get("metadata", {}).get("null_normalization") != "optional_empty_text_fields_use_null"
    ]

    return {
        "dataset": str(path.relative_to(ROOT)),
        "sample_count": len(samples),
        "status": {
            "expected_sample_count_1000": "pass" if len(samples) == 1000 else "fail",
            "factuality_context_complete": "pass" if not factuality_empty_context else "fail",
            "non_factuality_context_uses_null": "pass" if not non_factuality_context_not_null else "fail",
            "no_blank_strings": "pass" if not blank_string_locations else "fail",
            "field_contract_declared": "pass" if not field_contract_missing else "fail",
            "score_format_declared": "pass" if not score_format_missing else "fail",
            "split_leakage": "pass" if task_prompt_leaks(samples) == 0 else "fail",
            "content_duplicates": "pass" if content_duplicates(samples) == 0 else "fail",
        },
        "counts": {
            "factuality_total": len(factuality),
            "factuality_empty_context": len(factuality_empty_context),
            "non_factuality_context_not_null": len(non_factuality_context_not_null),
            "blank_string_count": len(blank_string_locations),
            "field_contract_missing": len(field_contract_missing),
            "score_format_missing": len(score_format_missing),
            "task_prompt_split_leakage": task_prompt_leaks(samples),
            "content_duplicates": content_duplicates(samples),
        },
        "review_items": {
            "blank_string_examples": blank_string_locations[:20],
            "field_contract_missing": field_contract_missing[:20],
            "non_factuality_context_not_null": non_factuality_context_not_null[:20],
        },
    }


def main() -> None:
    report = {
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "core": audit_core(),
        "chinese": audit_chinese(),
        "closure_criteria": {
            "critical_issues_pass": [
                "c1_factuality_null_human_label",
                "c1_factuality_empty_answer_b",
                "c1_single_answer_marker",
                "c2_score_format_declared",
                "c3_core1200_subset_of_core2400",
            ],
            "review_items_allowed_with_report": [
                "phone_like_candidates",
                "short_non_factuality_answers",
            "open_qa_repeated_prompt_count",
        ],
        "chinese_dataset_required_pass": [
            "expected_sample_count_1000",
            "factuality_context_complete",
            "non_factuality_context_uses_null",
            "no_blank_strings",
            "field_contract_declared",
            "score_format_declared",
        ],
    },
    }
    write_path = DATASETS / "compliance_audit_report.json"
    write_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["core"]["status"], ensure_ascii=False, indent=2))
    print(json.dumps(report["chinese"]["status"], ensure_ascii=False, indent=2))
    print(write_path)


if __name__ == "__main__":
    main()
