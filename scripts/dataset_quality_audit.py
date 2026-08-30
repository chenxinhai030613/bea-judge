"""
Dataset quality audit for BEA-Judge sample datasets.

The audit is intentionally strict against the DOCX construction plan while
also reporting the implemented flat schema used by the current JSON files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
PROCESSED = DATASETS / "processed"
REPORT_JSON = DATASETS / "dataset_quality_audit_detailed.json"
REPORT_MD = DATASETS / "dataset_quality_audit_report.md"

MISSING_THRESHOLD = 0.05
EXPECTED_TOP_LEVEL_TYPES = {
    "id": str,
    "dataset": str,
    "task_type": str,
    "prompt": str,
    "context": (str, type(None)),
    "answer_a": str,
    "answer_b": (str, type(None)),
    "reference": (str, type(None)),
    "human_score": dict,
    "human_label": (str, type(None)),
    "language": str,
    "split": str,
    "metadata": dict,
}
DOCX_NESTED_REQUIRED = ["id", "source", "task", "input", "answers", "label", "quality", "metadata"]
TASK_TYPES = {"open_qa", "pairwise_bias", "factuality_rag"}
LANGUAGES = {"en", "zh"}
SPLITS = {"train", "dev", "test"}
PAIRWISE_LABELS = {"A>B", "B>A", "Tie"}
FACTUALITY_LABELS = {"supported", "unsupported", "ambiguous"}
VALID_LABELS = PAIRWISE_LABELS | FACTUALITY_LABELS
SCORE_RANGE_KEYS = {
    "pairwise_preference": (-1.0, 1.0),
    "factuality_label_score": (0.0, 1.0),
    "factuality_score_0_1": (0.0, 1.0),
    "factuality_score_1_5": (1.0, 5.0),
    "faithfulness_a_1_5": (1.0, 5.0),
    "faithfulness_b_1_5": (1.0, 5.0),
}
DIMENSION_SCORE_KEYS = {"relevance", "completeness", "factuality", "instruction_following", "clarity", "safety"}
ID_RE = re.compile(r"^[A-Za-z0-9_./:-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_files(processed_dir: Path) -> List[Path]:
    return sorted(path for path in processed_dir.glob("*.json") if path.is_file())


def read_samples(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return [row for row in payload["samples"] if isinstance(row, dict)], payload.get("dataset_info", {})
    raise ValueError(f"JSON file does not contain a samples list: {path}")


def is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                nested = flatten(child, child_prefix)
                if nested:
                    out.update(nested)
                else:
                    out[child_prefix] = child
            else:
                out[child_prefix] = child
        return out
    return {prefix: value}


def missing_kind(value: Any) -> str | None:
    if value is None:
        return "null"
    if is_nan(value):
        return "nan"
    if isinstance(value, str):
        if value == "":
            return "empty_string"
        if not value.strip():
            return "whitespace_only"
    return None


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def audit_completeness(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    flattened_rows = [flatten(row) for row in samples]
    fields = sorted(set().union(*(row.keys() for row in flattened_rows)) if flattened_rows else [])
    row_count = len(samples)
    field_reports: Dict[str, Dict[str, Any]] = {}

    for field in fields:
        counts = Counter()
        type_counts = Counter()
        for row in flattened_rows:
            if field not in row:
                counts["absent"] += 1
                continue
            value = row[field]
            kind = missing_kind(value)
            if kind:
                counts[kind] += 1
            type_counts[type_name(value)] += 1

        missing_total = sum(counts[k] for k in ("absent", "null", "nan", "empty_string", "whitespace_only"))
        field_reports[field] = {
            "absent": counts["absent"],
            "null": counts["null"],
            "nan": counts["nan"],
            "empty_string": counts["empty_string"],
            "whitespace_only": counts["whitespace_only"],
            "missing_total": missing_total,
            "missing_ratio": round(missing_total / row_count, 6) if row_count else 0.0,
            "types": dict(type_counts),
            "over_threshold": (missing_total / row_count) > MISSING_THRESHOLD if row_count else False,
        }
    return {"row_count": row_count, "field_count": len(fields), "fields": field_reports}


def sample_source_id(sample: Dict[str, Any]) -> str:
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return str(
        metadata.get("source")
        or metadata.get("source_url")
        or sample.get("dataset")
        or "unknown"
    )


def add_issue(issues: List[Dict[str, Any]], category: str, severity: str, item: str, evidence: str, recommendation: str) -> None:
    issues.append(
        {
            "category": category,
            "severity": severity,
            "item": item,
            "evidence": evidence,
            "recommendation": recommendation,
        }
    )


def validate_samples(path: Path, samples: List[Dict[str, Any]], info: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    completeness = audit_completeness(samples)
    row_count = len(samples)

    for field in DOCX_NESTED_REQUIRED:
        missing = sum(1 for sample in samples if field not in sample)
        if missing:
            add_issue(
                issues,
                "schema_consistency",
                "high" if field in {"source", "task", "input", "answers", "label", "quality"} else "medium",
                f"missing_docx_nested_field:{field}",
                f"{missing}/{row_count} samples do not contain DOCX nested field `{field}`.",
                "Either export a canonical nested view matching the DOCX schema, or update the DOCX to explicitly accept the implemented flat schema.",
            )

    missing_required_top = {
        key: sum(1 for sample in samples if key not in sample or missing_kind(sample.get(key)) is not None)
        for key in ("id", "dataset", "task_type", "prompt", "answer_a", "human_score", "human_label", "language", "split", "metadata")
    }
    for key, count in missing_required_top.items():
        if count:
            add_issue(
                issues,
                "completeness",
                "high",
                f"required_top_level_missing:{key}",
                f"{count}/{row_count} samples have missing `{key}` under implemented schema.",
                f"Regenerate or repair rows with non-empty `{key}` before training/evaluation.",
            )

    type_mismatches: Counter[str] = Counter()
    enum_errors: Counter[str] = Counter()
    id_format_errors = 0
    score_range_errors: List[Dict[str, Any]] = []
    task_contract_errors: Counter[str] = Counter()
    source_counts = Counter()

    for sample in samples:
        source_counts[sample_source_id(sample)] += 1
        for field, expected in EXPECTED_TOP_LEVEL_TYPES.items():
            if field in sample and not isinstance(sample.get(field), expected):
                type_mismatches[field] += 1

        sample_id = sample.get("id")
        if isinstance(sample_id, str) and not ID_RE.match(sample_id):
            id_format_errors += 1

        task_type = sample.get("task_type")
        if task_type not in TASK_TYPES:
            enum_errors["task_type"] += 1
        if sample.get("language") not in LANGUAGES:
            enum_errors["language"] += 1
        if sample.get("split") not in SPLITS:
            enum_errors["split"] += 1
        label = sample.get("human_label")
        if label is not None and label not in VALID_LABELS:
            enum_errors["human_label"] += 1

        if task_type == "factuality_rag" and missing_kind(sample.get("context")) is not None:
            task_contract_errors["factuality_missing_context"] += 1
        if task_type in {"open_qa", "pairwise_bias"} and missing_kind(sample.get("answer_b")) is not None:
            task_contract_errors["pairwise_missing_answer_b"] += 1

        human_score = sample.get("human_score")
        if isinstance(human_score, dict):
            for key, (lo, hi) in SCORE_RANGE_KEYS.items():
                value = human_score.get(key)
                if value is not None:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        score_range_errors.append({"id": sample_id, "field": key, "value": value, "reason": "not_numeric"})
                        continue
                    if numeric < lo or numeric > hi:
                        score_range_errors.append({"id": sample_id, "field": key, "value": numeric, "range": [lo, hi]})
            dims = human_score.get("dimension_scores_1_5")
            if isinstance(dims, dict):
                for key, value in dims.items():
                    if key in DIMENSION_SCORE_KEYS:
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            score_range_errors.append({"id": sample_id, "field": f"dimension_scores_1_5.{key}", "value": value, "reason": "not_numeric"})
                            continue
                        if numeric < 1 or numeric > 5:
                            score_range_errors.append({"id": sample_id, "field": f"dimension_scores_1_5.{key}", "value": numeric, "range": [1, 5]})

    for field, count in type_mismatches.items():
        add_issue(
            issues,
            "type_consistency",
            "high" if field in {"id", "task_type", "prompt", "answer_a", "human_score"} else "medium",
            f"type_mismatch:{field}",
            f"{count}/{row_count} samples have unexpected type for `{field}`.",
            f"Normalize `{field}` to the declared type before model input construction.",
        )

    for field, count in enum_errors.items():
        add_issue(
            issues,
            "value_consistency",
            "high",
            f"enum_error:{field}",
            f"{count}/{row_count} samples contain values outside the allowed enum for `{field}`.",
            "Map labels/languages/splits to the documented enum or exclude invalid rows.",
        )

    if id_format_errors:
        add_issue(
            issues,
            "format_consistency",
            "medium",
            "id_format",
            f"{id_format_errors}/{row_count} ids do not match the conservative identifier pattern.",
            "Normalize identifiers to ASCII-safe stable ids, preserving original id in metadata.",
        )

    for field, count in task_contract_errors.items():
        add_issue(
            issues,
            "task_contract",
            "high",
            field,
            f"{count}/{row_count} samples violate required task field contract.",
            "Repair task-specific required fields or remove affected rows from train/dev/test.",
        )

    if score_range_errors:
        add_issue(
            issues,
            "value_range",
            "high",
            "human_score_range",
            f"{len(score_range_errors)} score values are outside documented numeric ranges or non-numeric.",
            "Clamp only if caused by deterministic mapping error; otherwise re-map from raw labels and keep a correction log.",
        )

    over_threshold_fields = [
        {"field": field, **report}
        for field, report in completeness["fields"].items()
        if report["over_threshold"]
    ]

    return {
        "path": str(path.relative_to(ROOT)),
        "dataset_info": info,
        "sample_count": row_count,
        "source_counts": dict(source_counts),
        "completeness": completeness,
        "over_threshold_fields": over_threshold_fields[:80],
        "over_threshold_field_count": len(over_threshold_fields),
        "issues": issues,
        "type_mismatches": dict(type_mismatches),
        "enum_errors": dict(enum_errors),
        "task_contract_errors": dict(task_contract_errors),
        "score_range_error_examples": score_range_errors[:20],
    }


def duplicates(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = Counter()
    by_content = Counter()
    for sample in samples:
        by_id[str(sample.get("id"))] += 1
        by_content[
            (
                sample.get("task_type"),
                sample.get("prompt"),
                sample.get("answer_a"),
                sample.get("answer_b"),
            )
        ] += 1
    return {
        "duplicate_id_count": sum(count - 1 for count in by_id.values() if count > 1),
        "duplicate_content_count": sum(count - 1 for count in by_content.values() if count > 1),
        "duplicate_id_examples": [key for key, count in by_id.items() if count > 1][:20],
    }


def split_leaks(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Dict[Tuple[Any, Any], set] = defaultdict(set)
    for sample in samples:
        key = (sample.get("task_type"), sample.get("prompt"))
        by_key[key].add(sample.get("split"))
    leaks = [key for key, splits in by_key.items() if len(splits) > 1]
    return {"task_prompt_leakage_count": len(leaks), "examples": [str(key)[:180] for key in leaks[:10]]}


def aggregate_primary_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    primary_paths = [
        "datasets/processed/bea_judge_core_2400.json",
        "datasets/processed/chinese_professional_annotated_1000.json",
    ]
    primary_path_set = {path.replace("/", "\\") for path in primary_paths}
    samples: List[Dict[str, Any]] = []
    for report in reports:
        if report["path"].replace("/", "\\") in primary_path_set:
            path = ROOT / report["path"]
            rows, _ = read_samples(path)
            samples.extend(rows)

    return {
        "primary_scope": primary_paths,
        "sample_count": len(samples),
        "by_task_type": dict(Counter(sample.get("task_type") for sample in samples)),
        "by_language": dict(Counter(sample.get("language") for sample in samples)),
        "by_split": dict(Counter(sample.get("split") for sample in samples)),
        "by_dataset": dict(Counter(sample.get("dataset") for sample in samples)),
        "duplicates": duplicates(samples),
        "split_leakage": split_leaks(samples),
    }


def severity_counts(reports: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for report in reports:
        for issue in report["issues"]:
            counts[issue["severity"]] += 1
    return dict(counts)


def issue_table(reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in reports:
        for issue in report["issues"]:
            row = {"dataset": report["path"], **issue}
            rows.append(row)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(rows, key=lambda row: (severity_order.get(row["severity"], 9), row["dataset"], row["item"]))


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("# BEA-Judge 数据集质量审计报告")
    lines.append("")
    lines.append(f"- 生成时间：{report['created_at']}")
    lines.append(f"- 审计路径：`{report['datasets_root']}`")
    lines.append(f"- 缺失阈值：>{MISSING_THRESHOLD:.0%}")
    lines.append(f"- 主审计范围：`bea_judge_core_2400.json` + `chinese_professional_annotated_1000.json`")
    lines.append("")
    agg = report["aggregate_primary"]
    lines.append("## 总览")
    lines.append("")
    lines.append(f"- 主范围样本数：{agg['sample_count']}")
    lines.append(f"- 任务分布：`{agg['by_task_type']}`")
    lines.append(f"- 语言分布：`{agg['by_language']}`")
    lines.append(f"- split 分布：`{agg['by_split']}`")
    lines.append(f"- 重复 ID：{agg['duplicates']['duplicate_id_count']}；重复内容：{agg['duplicates']['duplicate_content_count']}")
    lines.append(f"- task+prompt 跨 split 泄漏：{agg['split_leakage']['task_prompt_leakage_count']}")
    lines.append("")
    lines.append("## 高优先级问题")
    lines.append("")
    high_rows = [row for row in report["issues"] if row["severity"] == "high"]
    if not high_rows:
        lines.append("- 未发现高风险问题。")
    else:
        for row in high_rows[:40]:
            lines.append(f"- `{row['dataset']}` `{row['item']}`：{row['evidence']} 建议：{row['recommendation']}")
    lines.append("")
    lines.append("## 数据集逐项摘要")
    lines.append("")
    for ds in report["dataset_reports"]:
        lines.append(f"### `{ds['path']}`")
        lines.append(f"- 样本数：{ds['sample_count']}；字段数：{ds['completeness']['field_count']}；超过 5% 缺失阈值字段数：{ds['over_threshold_field_count']}")
        lines.append(f"- 类型错误：`{ds['type_mismatches']}`；枚举错误：`{ds['enum_errors']}`；任务契约错误：`{ds['task_contract_errors']}`")
        if ds["over_threshold_fields"]:
            preview = [
                {
                    "field": field["field"],
                    "missing_ratio": field["missing_ratio"],
                    "absent": field["absent"],
                    "null": field["null"],
                    "empty_string": field["empty_string"],
                    "whitespace_only": field["whitespace_only"],
                }
                for field in ds["over_threshold_fields"][:8]
            ]
            lines.append(f"- 缺失字段示例：`{preview}`")
        lines.append("")
    lines.append("## 清洗建议优先级")
    lines.append("")
    lines.append("1. 先统一文档规范与 JSON 产物：选择嵌套 canonical schema，或在 DOCX 中正式承认当前 flat schema。")
    lines.append("2. 对核心公开数据执行空字符串规范化：非必需 `context/reference` 改为 `null`，并写入 `metadata.missing_reason`。")
    lines.append("3. 保留 `chinese_professional_annotated_1000.json` 作为中文主版本，归档旧 300 条版本或标注为 legacy。")
    lines.append("4. 对 `splits_zh` 重新核验同一 parent/prompt 不跨 train/dev/test；当前按 task+prompt 粗粒度检查存在泄漏风险信号。")
    lines.append("5. 建立每次构建后的自动门禁：样本数、字段类型、枚举、空字符串、重复、split 泄漏、score range 全部通过后再训练。")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BEA-Judge JSON sample datasets.")
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    args = parser.parse_args()

    processed_dir = args.datasets / "processed"
    reports = []
    for path in dataset_files(processed_dir):
        samples, info = read_samples(path)
        reports.append(validate_samples(path, samples, info))

    report = {
        "created_at": utc_now(),
        "datasets_root": str(args.datasets.resolve()),
        "reference_docx": str((ROOT / "BEA-Judge模型构建全流程方案_期刊课题版.docx").resolve()),
        "missing_threshold": MISSING_THRESHOLD,
        "dataset_reports": reports,
        "aggregate_primary": aggregate_primary_reports(reports),
        "severity_counts": severity_counts(reports),
        "issues": issue_table(reports),
        "notes": [
            "DOCX reference currently defines a nested schema, while JSON datasets implement a flat schema.",
            "Processed split files are derivative and not counted in the primary aggregate to avoid double counting.",
        ],
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, REPORT_MD)
    print(json.dumps({
        "reports": len(reports),
        "severity_counts": report["severity_counts"],
        "aggregate_primary": report["aggregate_primary"],
        "json": str(REPORT_JSON),
        "markdown": str(REPORT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
