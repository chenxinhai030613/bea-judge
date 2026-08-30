"""Build a license-gated BEA-Judge-10K expansion dataset.

The builder is intentionally conservative: it only admits records from sources
with explicit redistribution-compatible licences and enough fields to construct
the existing BEA-Judge flat training schema. Missing external source files
produce an auditable readiness report instead of synthetic filler records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_adapter import samples_from_payload  # noqa: E402


DATASETS = ROOT / "datasets"
DEFAULT_EXISTING = DATASETS / "processed" / "bea_judge_cleaned_3400.json"
DEFAULT_SOURCE_DIR = DATASETS / "raw_v2"
DEFAULT_SPLIT_DIR = DATASETS / "splits_v2"
DEFAULT_PROCESSED = DATASETS / "processed" / "bea_judge_cleaned_10000.json"
DEFAULT_MANIFEST = DATASETS / "data_manifest_v2.json"
DEFAULT_REPORT = DATASETS / "expansion_v2_report.json"
DEFAULT_DA = DATASETS / "data_availability_v2_draft.md"
DEFAULT_SOURCE_METADATA = DEFAULT_SOURCE_DIR / "source_metadata.json"

PAIRWISE_LABELS = {"A>B", "B>A", "Tie"}
FACTUALITY_LABELS = {"supported", "unsupported", "ambiguous"}
VALID_LABELS = PAIRWISE_LABELS | FACTUALITY_LABELS
OPEN_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc0-1.0",
    "internal/self-built",
}
RESTRICTED_LICENSE_MARKERS = ("unknown", "non-commercial", "cc-by-nc", "research only", "no redistribution")


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class SourceSpec:
    key: str
    dataset: str
    task_family: str
    filename: str
    url: str
    license: str
    version: str
    target: int
    license_url: str = ""
    redistribution_allowed: bool = True
    external_eval_only: bool = False


SOURCE_SPECS = [
    SourceSpec(
        key="helpsteer2",
        dataset="helpsteer2",
        task_family="open_qa",
        filename="helpsteer2.jsonl",
        url="https://huggingface.co/datasets/nvidia/HelpSteer2",
        license="CC-BY-4.0",
        version="main",
        target=2000,
        license_url="https://huggingface.co/datasets/nvidia/HelpSteer2",
    ),
    SourceSpec(
        key="oasst1",
        dataset="oasst1",
        task_family="open_qa",
        filename="oasst1.jsonl",
        url="https://huggingface.co/datasets/OpenAssistant/oasst1",
        license="Apache-2.0",
        version="main",
        target=800,
        license_url="https://huggingface.co/datasets/OpenAssistant/oasst1",
    ),
    SourceSpec(
        key="offsetbias",
        dataset="offsetbias",
        task_family="pairwise_bias",
        filename="offsetbias.jsonl",
        url="https://github.com/ncsoft/offsetbias",
        license="BSD-3-Clause",
        version="main",
        target=1500,
        license_url="https://huggingface.co/datasets/NCSOFT/offsetbias",
    ),
    SourceSpec(
        key="ragtruth",
        dataset="ragtruth",
        task_family="factuality_rag",
        filename="ragtruth.jsonl",
        url="https://github.com/ParticleMedia/RAGTruth",
        license="MIT",
        version="main",
        target=2500,
        license_url="https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE",
    ),
    SourceSpec(
        key="rewardbench",
        dataset="rewardbench_external_eval",
        task_family="external_eval",
        filename="rewardbench.jsonl",
        url="https://huggingface.co/datasets/allenai/reward-bench",
        license="mixed-subset-license",
        version="main",
        target=600,
        license_url="https://huggingface.co/datasets/allenai/reward-bench",
        redistribution_allowed=False,
        external_eval_only=True,
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any, max_chars: Optional[int] = None) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).replace("\u0000", " ").split())
    if max_chars is not None:
        text = text[:max_chars].strip()
    return text


def stable_hash(*parts: Any, length: int = 16) -> str:
    joined = "\n".join(normalize_text(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_license(value: Any) -> str:
    return normalize_text(value).lower()


def license_gate(spec: SourceSpec) -> Tuple[bool, str]:
    licence = canonical_license(spec.license)
    if spec.external_eval_only:
        return False, "external_eval_only"
    if not spec.redistribution_allowed:
        return False, "redistribution_not_allowed"
    if not licence or any(marker in licence for marker in RESTRICTED_LICENSE_MARKERS):
        return False, "license_missing_or_restricted"
    if licence not in OPEN_LICENSES:
        return False, "license_not_allowlisted"
    return True, "ok"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_records(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            lines = list(handle)
        for line in lines:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
        return rows
    if suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        for key in ("data", "records", "samples", "train"):
            if isinstance(payload, dict) and isinstance(payload.get(key), list):
                return [row for row in payload[key] if isinstance(row, dict)]
        return [payload] if isinstance(payload, dict) else []
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"unsupported source format: {path}")


def first_text(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (list, tuple)):
            value = "\n".join(str(item) for item in value)
        text = normalize_text(value)
        if text:
            return text
    return ""


def first_number(record: Dict[str, Any], keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        value = record.get(key)
        try:
            if value is not None and str(value).strip() != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def detect_language(*texts: str) -> str:
    joined = "".join(texts)
    zh = sum(1 for char in joined if "\u4e00" <= char <= "\u9fff")
    return "zh" if zh >= 6 else "en"


def pairwise_score(label: str) -> float:
    return {"A>B": 1.0, "B>A": -1.0, "Tie": 0.0}[label]


def factuality_score(label: str) -> float:
    return {"supported": 1.0, "ambiguous": 0.5, "unsupported": 0.0}[label]


def label_from_text(value: Any) -> Optional[str]:
    text = normalize_text(value).lower()
    if text in {"a>b", "a", "chosen", "preferred", "response_a", "answer_a"}:
        return "A>B"
    if text in {"b>a", "b", "rejected", "response_b", "answer_b"}:
        return "B>A"
    if text in {"tie", "draw", "equal", "same"}:
        return "Tie"
    if text in {"supported", "true", "faithful", "correct", "0"}:
        return "supported"
    if text in {"unsupported", "false", "hallucinated", "incorrect", "1"}:
        return "unsupported"
    if text in {"ambiguous", "uncertain", "partial", "mixed"}:
        return "ambiguous"
    return None


def pairwise_label_from_text(value: Any) -> Optional[str]:
    text = normalize_text(value).lower()
    if text in {"a>b", "a", "1", "1.0", "chosen", "preferred", "response_a", "answer_a", "output_1"}:
        return "A>B"
    if text in {"b>a", "b", "2", "2.0", "rejected", "response_b", "answer_b", "output_2"}:
        return "B>A"
    if text in {"tie", "draw", "equal", "same", "both", "neither"}:
        return "Tie"
    return None


def make_sample(
    *,
    source: SourceSpec,
    task_type: str,
    prompt: str,
    answer_a: str,
    label: str,
    context: str = "",
    answer_b: str = "",
    reference: str = "",
    source_record_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or {})
    metadata.update(
        {
            "source": source.key,
            "source_url": source.url,
            "source_record_id": source_record_id,
            "license": source.license,
            "source_version": source.version,
            "preprocessing_schema_version": "BEA-Judge-10K-v2",
            "score_format": "single_answer_factuality" if label in FACTUALITY_LABELS else "pairwise_preference",
            "scoring_system": "single_answer_factuality" if label in FACTUALITY_LABELS else "pairwise_preference",
        }
    )
    human_score = {
        "score_format": metadata["score_format"],
        "scoring_system": metadata["scoring_system"],
        "label": label,
    }
    if label in PAIRWISE_LABELS:
        human_score["pairwise_preference"] = pairwise_score(label)
    else:
        human_score["factuality_label_score"] = factuality_score(label)
        human_score["factuality_score_0_1"] = factuality_score(label)
    sample_id = f"{source.dataset}_{stable_hash(source_record_id or '', prompt, answer_a, answer_b, label)}"
    return {
        "id": sample_id,
        "dataset": source.dataset,
        "task_type": task_type,
        "prompt": normalize_text(prompt, 4000),
        "context": normalize_text(context, 8000),
        "answer_a": normalize_text(answer_a, 5000),
        "answer_b": normalize_text(answer_b, 5000),
        "reference": normalize_text(reference, 4000),
        "human_score": human_score,
        "human_label": label,
        "language": detect_language(prompt, answer_a, answer_b, context),
        "metadata": metadata,
    }


def valid_flat_sample(sample: Dict[str, Any]) -> Tuple[bool, str]:
    for key in ("id", "dataset", "task_type", "prompt", "answer_a", "human_score", "human_label"):
        if sample.get(key) in (None, ""):
            return False, f"missing_{key}"
    if sample.get("human_label") not in VALID_LABELS:
        return False, "invalid_label"
    if sample.get("task_type") in {"open_qa", "pairwise_bias"} and not sample.get("answer_b"):
        return False, "missing_answer_b"
    if sample.get("task_type") == "factuality_rag" and not sample.get("context"):
        return False, "missing_context"
    return True, "ok"


def adapt_helpsteer2(records: Sequence[Dict[str, Any]], spec: SourceSpec, limit: int) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Tuple[float, Dict[str, Any]]]] = defaultdict(list)
    for row in records:
        prompt = first_text(row, ("prompt", "instruction", "question"))
        answer = first_text(row, ("response", "answer", "output", "completion"))
        score = first_number(row, ("overall", "helpfulness", "helpful", "score", "rating"))
        if prompt and answer and score is not None:
            grouped[prompt].append((score, row))
    samples: List[Dict[str, Any]] = []
    for prompt, rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        low_score, low = min(rows, key=lambda item: item[0])
        high_score, high = max(rows, key=lambda item: item[0])
        if high_score == low_score:
            high = rows[0][1]
            low = rows[1][1]
            label = "Tie"
        else:
            label = "A>B"
        answer_a = first_text(high, ("response", "answer", "output", "completion"))
        answer_b = first_text(low, ("response", "answer", "output", "completion"))
        source_id = first_text(high, ("id", "sample_id", "conversation_id")) or stable_hash(prompt)
        samples.append(
            make_sample(
                source=spec,
                task_type="open_qa",
                prompt=prompt,
                answer_a=answer_a,
                answer_b=answer_b,
                label=label,
                source_record_id=source_id,
                metadata={"score_gap": high_score - low_score},
            )
        )
        if len(samples) >= limit:
            break
    return samples


def adapt_preference_pairs(records: Sequence[Dict[str, Any]], spec: SourceSpec, limit: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in records:
        prompt = first_text(row, ("prompt", "instruction", "question", "context"))
        answer_a = first_text(row, ("answer_a", "response_a", "chosen", "better", "output_1"))
        answer_b = first_text(row, ("answer_b", "response_b", "rejected", "worse", "output_2"))
        label = pairwise_label_from_text(first_text(row, ("label", "winner", "preference", "chosen_label"))) or "A>B"
        if label not in PAIRWISE_LABELS:
            continue
        if prompt and answer_a and answer_b:
            source_id = first_text(row, ("id", "sample_id", "pair_id", "conversation_id")) or stable_hash(prompt, answer_a)
            task_type = "pairwise_bias" if spec.task_family == "pairwise_bias" else "open_qa"
            samples.append(
                make_sample(
                    source=spec,
                    task_type=task_type,
                    prompt=prompt,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    label=label,
                    source_record_id=source_id,
                    metadata={
                        "bias_type": first_text(row, ("bias_type", "perturbation", "bias_group")) or None,
                    },
                )
            )
        if len(samples) >= limit:
            break
    return samples


def adapt_ragtruth(records: Sequence[Dict[str, Any]], spec: SourceSpec, limit: int) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in records:
        prompt = first_text(row, ("prompt", "question", "query", "instruction"))
        context = first_text(row, ("context", "passage", "document", "source", "evidence"))
        answer = first_text(row, ("answer", "response", "output", "completion"))
        label = label_from_text(first_text(row, ("label", "factuality", "hallucination", "is_hallucinated")))
        if not label:
            hallucination = first_number(row, ("hallucination", "is_hallucinated", "hallucination_label"))
            if hallucination is not None:
                label = "unsupported" if hallucination >= 0.5 else "supported"
        if prompt and context and answer and label in FACTUALITY_LABELS:
            source_id = first_text(row, ("id", "sample_id", "response_id", "question_id")) or stable_hash(prompt, answer)
            samples.append(
                make_sample(
                    source=spec,
                    task_type="factuality_rag",
                    prompt=prompt,
                    context=context,
                    answer_a=answer,
                    answer_b="[SINGLE_ANSWER_FACTUALITY_TASK]",
                    reference=first_text(row, ("reference", "ground_truth", "gold_answer")),
                    label=label,
                    source_record_id=source_id,
                    metadata={
                        "original_ragtruth_record": row,
                        "label_scope": "response_level_hallucination",
                        "source_label_schema": "ragtruth_binary",
                        "label_granularity_warning": False,
                    },
                )
            )
        if len(samples) >= limit:
            break
    return samples


def adapt_source(records: Sequence[Dict[str, Any]], spec: SourceSpec) -> List[Dict[str, Any]]:
    if spec.key == "helpsteer2":
        return adapt_helpsteer2(records, spec, spec.target)
    if spec.key == "ragtruth":
        return adapt_ragtruth(records, spec, spec.target)
    if spec.key in {"offsetbias", "oasst1"}:
        return adapt_preference_pairs(records, spec, spec.target)
    return []


def content_key(sample: Dict[str, Any]) -> str:
    return stable_hash(
        sample.get("task_type"),
        sample.get("prompt"),
        sample.get("answer_a"),
        sample.get("answer_b"),
        length=32,
    )


def deduplicate(samples: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seen_ids = set()
    seen_content = set()
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("id"))
        key = content_key(sample)
        if sample_id in seen_ids:
            rejected.append({"id": sample_id, "reason": "duplicate_id"})
            continue
        if key in seen_content:
            rejected.append({"id": sample_id, "reason": "duplicate_content"})
            continue
        seen_ids.add(sample_id)
        seen_content.add(key)
        kept.append(sample)
    return kept, rejected


def assign_split(sample: Dict[str, Any]) -> str:
    value = int(stable_hash(sample.get("id"), sample.get("prompt"), length=8), 16) % 100
    if value < 70:
        return "train"
    if value < 85:
        return "dev"
    return "test"


def write_splits(samples: Sequence[Dict[str, Any]], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        rows = [sample for sample in samples if sample.get("split") == split]
        write_json(
            split_dir / f"{split}.json",
            {
                "dataset_info": {
                    "name": f"BEA-Judge-10K v2 {split}",
                    "created_at": utc_now(),
                    "schema": "BEA-Judge flat split schema",
                    "sample_count": len(rows),
                },
                "samples": rows,
            },
        )


def load_source_metadata(source_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = source_dir / "source_metadata.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    sources = payload.get("sources", {}) if isinstance(payload, dict) else {}
    if not isinstance(sources, dict):
        return {}
    return {str(key): value for key, value in sources.items() if isinstance(value, dict)}


def build_manifest(source_dir: Path, source_reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    source_metadata = load_source_metadata(source_dir)
    sources: Dict[str, Any] = {}
    for spec in SOURCE_SPECS:
        path = source_dir / spec.filename
        allowed, reason = license_gate(spec)
        meta = source_metadata.get(spec.key, {})
        acquisition_date = meta.get("acquisition_date") or meta.get("acquired_at")
        if not acquisition_date and path.exists():
            acquisition_date = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        sha256 = meta.get("sha256") or (sha256_file(path) if path.exists() else None)
        sources[spec.key] = {
            "url": spec.url,
            "path": path_relative_to_root(path),
            "acquisition_date": acquisition_date,
            "license": spec.license,
            "license_url": meta.get("license_url") or spec.license_url,
            "license_status": "present" if spec.license else "missing",
            "version": meta.get("revision") or meta.get("version") or spec.version,
            "sha256": sha256,
            "record_count": meta.get("record_count"),
            "redistribution_allowed": spec.redistribution_allowed,
            "external_eval_only": spec.external_eval_only,
            "admission_allowed": allowed,
            "admission_reason": reason,
            "preprocessing_schema_version": "BEA-Judge-10K-v2",
        }
    return {
        "created_at": utc_now(),
        "source_dir": path_relative_to_root(source_dir),
        "sources": sources,
        "source_reports": list(source_reports),
    }


def load_existing_samples(path: Path) -> List[Dict[str, Any]]:
    samples = samples_from_payload(read_json(path))
    for sample in samples:
        sample.setdefault("metadata", {})
        sample["metadata"].setdefault("expansion_origin", "existing_bea_judge_3400")
    return samples


def load_expansion_samples(source_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    samples: List[Dict[str, Any]] = []
    reports: List[Dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        allowed, reason = license_gate(spec)
        path = source_dir / spec.filename
        report = {
            "source": spec.key,
            "dataset": spec.dataset,
            "path": path_relative_to_root(path),
            "target": spec.target,
            "license": spec.license,
            "admission_allowed": allowed,
            "admission_reason": reason,
            "file_exists": path.exists(),
            "raw_records": 0,
            "accepted_records": 0,
            "rejected_records": 0,
        }
        if not allowed or not path.exists():
            reports.append(report)
            continue
        raw = read_records(path)
        adapted = adapt_source(raw, spec)
        accepted: List[Dict[str, Any]] = []
        for sample in adapted:
            ok, sample_reason = valid_flat_sample(sample)
            if ok:
                sample["metadata"]["expansion_origin"] = spec.key
                accepted.append(sample)
        report["raw_records"] = len(raw)
        report["accepted_records"] = len(accepted)
        report["rejected_records"] = len(adapted) - len(accepted)
        samples.extend(accepted)
        reports.append(report)
    return samples, reports


def split_leakage(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        groups[content_key(sample)].add(str(sample.get("split")))
    leaking = [key for key, splits in groups.items() if len(splits) > 1]
    return {"content_group_count": len(groups), "cross_split_duplicate_content": len(leaking)}


def statistics(samples: Sequence[Dict[str, Any]], rejected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    pairwise = [sample for sample in samples if sample.get("human_label") in PAIRWISE_LABELS]
    tie_count = sum(1 for sample in pairwise if sample.get("human_label") == "Tie")
    return {
        "total_samples": len(samples),
        "by_task_type": dict(Counter(str(sample.get("task_type")) for sample in samples)),
        "by_dataset": dict(Counter(str(sample.get("dataset")) for sample in samples)),
        "by_split": dict(Counter(str(sample.get("split")) for sample in samples)),
        "by_language": dict(Counter(str(sample.get("language")) for sample in samples)),
        "human_label_distribution": dict(Counter(str(sample.get("human_label")) for sample in samples)),
        "pairwise_count": len(pairwise),
        "tie_ratio_among_pairwise": round(tie_count / len(pairwise), 4) if pairwise else 0.0,
        "rejected_count": len(rejected),
        "split_leakage": split_leakage(samples),
    }


def data_availability_draft(manifest: Dict[str, Any], report: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Data Availability Draft for BEA-Judge-10K v2",
            "",
            "The processed BEA-Judge-10K v2 dataset, source manifest, preprocessing scripts, checksums, split definitions, and model-output tables should be deposited in a DOI-issuing repository before submission. Reused public datasets must be cited separately and redistributed only when their source licences permit redistribution.",
            "",
            "## Included Processed Data",
            "",
            f"- Target records: {report['targets']['target_count']}.",
            f"- Built records: {report['statistics']['total_samples']}.",
            f"- Admission status: {report['gates']['target_count_gate']}.",
            "",
            "## Source Licence Audit",
            "",
            "| source | license | redistribution_allowed | admission_allowed | reason | sha256 |",
            "| --- | --- | --- | --- | --- | --- |",
            *[
                f"| {key} | {item['license']} | {item['redistribution_allowed']} | {item['admission_allowed']} | {item['admission_reason']} | {item['sha256'] or ''} |"
                for key, item in manifest["sources"].items()
            ],
            "",
            "## Missing Information / Risk Flags",
            "",
            "- Confirm final repository DOI/accession before manuscript submission.",
            "- Do not redistribute sources marked external_eval_only or redistribution_not_allowed.",
            "- If fewer than 9500 records are built, report BEA-Judge-10K as not yet ready for formal SCI results.",
            "",
        ]
    )


def build_expansion(
    *,
    existing_path: Path,
    source_dir: Path,
    output_path: Path,
    split_dir: Path,
    manifest_path: Path,
    report_path: Path,
    data_availability_path: Path,
    target_count: int,
    minimum_count: int,
    allow_incomplete: bool,
) -> Dict[str, Any]:
    existing = load_existing_samples(existing_path)
    expansion, source_reports = load_expansion_samples(source_dir)
    combined, rejected = deduplicate([*existing, *expansion])
    for sample in combined:
        if sample.get("split") not in {"train", "dev", "test"}:
            sample["split"] = assign_split(sample)

    stats = statistics(combined, rejected)
    manifest = build_manifest(source_dir, source_reports)
    gates = {
        "minimum_count_gate": stats["total_samples"] >= minimum_count,
        "target_count_gate": stats["total_samples"] >= target_count,
        "duplicate_id_or_content_zero": len(rejected) == 0,
        "cross_split_duplicate_content_zero": stats["split_leakage"]["cross_split_duplicate_content"] == 0,
        "license_metadata_complete": all(item.get("license") for item in manifest["sources"].values()),
        "factuality_context_missing_zero": all(
            sample.get("context") for sample in combined if sample.get("task_type") == "factuality_rag"
        ),
    }
    report = {
        "created_at": utc_now(),
        "targets": {"target_count": target_count, "minimum_count": minimum_count},
        "inputs": {
            "existing_dataset": path_relative_to_root(existing_path),
            "source_dir": path_relative_to_root(source_dir),
        },
        "statistics": stats,
        "source_reports": source_reports,
        "gates": gates,
        "rejected_records": rejected[:500],
    }
    write_json(manifest_path, manifest)
    write_json(report_path, report)
    data_availability_path.parent.mkdir(parents=True, exist_ok=True)
    data_availability_path.write_text(data_availability_draft(manifest, report), encoding="utf-8")

    if stats["total_samples"] >= minimum_count or allow_incomplete:
        write_splits(combined, split_dir)
        write_json(
            output_path,
            {
                "dataset_info": {
                    "name": "BEA-Judge-10K-v2",
                    "created_at": utc_now(),
                    "schema": "BEA-Judge flat training schema",
                    "sample_count": len(combined),
                    "source_manifest": path_relative_to_root(manifest_path),
                    "expansion_report": path_relative_to_root(report_path),
                },
                "samples": combined,
            },
        )
        report["outputs"] = {
            "processed_dataset": path_relative_to_root(output_path),
            "split_dir": path_relative_to_root(split_dir),
            "manifest": path_relative_to_root(manifest_path),
            "data_availability_draft": path_relative_to_root(data_availability_path),
        }
    else:
        report["outputs"] = {
            "processed_dataset": None,
            "split_dir": None,
            "manifest": path_relative_to_root(manifest_path),
            "data_availability_draft": path_relative_to_root(data_availability_path),
        }
        write_json(report_path, report)
        raise RuntimeError(
            f"BEA-Judge-10K v2 not ready: built {stats['total_samples']} records, "
            f"minimum required is {minimum_count}. Add vetted source files under {source_dir}."
        )
    write_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the BEA-Judge-10K v2 expansion dataset.")
    parser.add_argument("--existing", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-availability", type=Path, default=DEFAULT_DA)
    parser.add_argument("--target-count", type=int, default=10000)
    parser.add_argument("--minimum-count", type=int, default=9500)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_expansion(
        existing_path=args.existing,
        source_dir=args.source_dir,
        output_path=args.output,
        split_dir=args.split_dir,
        manifest_path=args.manifest,
        report_path=args.report,
        data_availability_path=args.data_availability,
        target_count=args.target_count,
        minimum_count=args.minimum_count,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps({"statistics": report["statistics"], "gates": report["gates"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
