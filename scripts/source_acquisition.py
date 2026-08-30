"""Acquire or audit raw BEA-Judge-10K v2 source files.

The script has two modes:
- default: try to export supported Hugging Face datasets to datasets/raw_v2
- --metadata-only: do not download; only record metadata for existing files

RAGTruth is kept as a manual/export source because its public GitHub release can
use multiple files. Place a builder-compatible ragtruth.jsonl in datasets/raw_v2,
then rerun this script with --metadata-only to record checksum metadata.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
DEFAULT_SOURCE_DIR = DATASETS / "raw_v2"
DEFAULT_METADATA = DEFAULT_SOURCE_DIR / "source_metadata.json"


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class AcquisitionSpec:
    key: str
    filename: str
    url: str
    license: str
    license_url: str
    revision: str
    hf_dataset: Optional[str] = None
    training_admission: bool = True
    external_eval_only: bool = False


SOURCE_CONFIGS: Dict[str, AcquisitionSpec] = {
    "helpsteer2": AcquisitionSpec(
        key="helpsteer2",
        filename="helpsteer2.jsonl",
        url="https://huggingface.co/datasets/nvidia/HelpSteer2",
        license="CC-BY-4.0",
        license_url="https://huggingface.co/datasets/nvidia/HelpSteer2",
        revision="main",
        hf_dataset="nvidia/HelpSteer2",
    ),
    "oasst1": AcquisitionSpec(
        key="oasst1",
        filename="oasst1.jsonl",
        url="https://huggingface.co/datasets/OpenAssistant/oasst1",
        license="Apache-2.0",
        license_url="https://huggingface.co/datasets/OpenAssistant/oasst1",
        revision="main",
        hf_dataset="OpenAssistant/oasst1",
    ),
    "offsetbias": AcquisitionSpec(
        key="offsetbias",
        filename="offsetbias.jsonl",
        url="https://huggingface.co/datasets/NCSOFT/offsetbias",
        license="BSD-3-Clause",
        license_url="https://huggingface.co/datasets/NCSOFT/offsetbias",
        revision="main",
        hf_dataset="NCSOFT/offsetbias",
    ),
    "ragtruth": AcquisitionSpec(
        key="ragtruth",
        filename="ragtruth.jsonl",
        url="https://github.com/ParticleMedia/RAGTruth",
        license="MIT",
        license_url="https://github.com/ParticleMedia/RAGTruth/blob/main/LICENSE",
        revision="main",
        hf_dataset=None,
    ),
    "rewardbench": AcquisitionSpec(
        key="rewardbench",
        filename="rewardbench.jsonl",
        url="https://huggingface.co/datasets/allenai/reward-bench",
        license="mixed-subset-license",
        license_url="https://huggingface.co/datasets/allenai/reward-bench",
        revision="main",
        hf_dataset="allenai/reward-bench",
        training_admission=False,
        external_eval_only=True,
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_existing_metadata(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], limit: Optional[int] = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def scalar_label(labels: Any, name: str) -> Optional[float]:
    if not isinstance(labels, dict):
        return None
    names = labels.get("name", [])
    values = labels.get("value", [])
    if not isinstance(names, list) or not isinstance(values, list):
        return None
    for index, item in enumerate(names):
        if item == name and index < len(values):
            try:
                return float(values[index])
            except (TypeError, ValueError):
                return None
    return None


def row_score(row: Dict[str, Any]) -> float:
    if row.get("rank") is not None:
        try:
            return -float(row["rank"])
        except (TypeError, ValueError):
            pass
    for key in ("overall", "helpfulness", "helpful", "score", "rating", "correctness", "coherence"):
        value = row.get(key)
        try:
            if value is not None and str(value).strip() != "":
                return float(value)
        except (TypeError, ValueError):
            continue
    quality = scalar_label(row.get("labels"), "quality")
    helpfulness = scalar_label(row.get("labels"), "helpfulness")
    values = [value for value in (quality, helpfulness) if value is not None]
    return sum(values) / len(values) if values else 0.0


def export_helpsteer2_rows(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for row in rows:
        out = dict(row)
        if "overall" not in out:
            out["overall"] = row_score(row)
        yield out


def export_offsetbias_rows(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    for row in rows:
        yield dict(row)


def export_oasst1_pairs(rows: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    prompts: Dict[str, Dict[str, Any]] = {}
    children: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if row.get("role") == "prompter":
            prompts[str(row.get("message_id"))] = row
        elif row.get("role") == "assistant" and row.get("parent_id"):
            children.setdefault(str(row.get("parent_id")), []).append(row)

    for parent_id in sorted(children):
        prompt = prompts.get(parent_id)
        replies = children[parent_id]
        if not prompt or len(replies) < 2:
            continue
        scored = [(row_score(reply), reply) for reply in replies if reply.get("text")]
        if len(scored) < 2:
            continue
        low_score, low = min(scored, key=lambda item: item[0])
        high_score, high = max(scored, key=lambda item: item[0])
        label = "Tie" if high_score == low_score else "A>B"
        yield {
            "id": f"{parent_id}:{high.get('message_id')}:{low.get('message_id')}",
            "prompt": prompt.get("text"),
            "answer_a": high.get("text"),
            "answer_b": low.get("text"),
            "label": label,
            "score_a": high_score,
            "score_b": low_score,
            "lang": prompt.get("lang"),
            "message_tree_id": prompt.get("message_tree_id"),
        }


def response_label(row: Dict[str, Any]) -> str:
    labels = row.get("labels")
    if isinstance(labels, list) and labels:
        return "unsupported"
    quality = str(row.get("quality", "")).lower()
    if quality in {"bad", "poor", "hallucinated"}:
        return "unsupported"
    return "supported"


def github_jsonl_lines(path: str) -> Iterator[Dict[str, Any]]:
    import requests

    meta_url = f"https://api.github.com/repos/ParticleMedia/RAGTruth/contents/{path}"
    meta = requests.get(meta_url, verify=False, timeout=60)
    meta.raise_for_status()
    blob = requests.get(meta.json()["git_url"], verify=False, timeout=180)
    blob.raise_for_status()
    payload = blob.json()
    text = base64.b64decode(payload["content"]).decode("utf-8")
    for line in text.splitlines():
        if line.strip():
            yield json.loads(line)


def export_ragtruth_rows() -> Iterator[Dict[str, Any]]:
    sources = {str(row.get("source_id")): row for row in github_jsonl_lines("dataset/source_info.jsonl")}
    for row in github_jsonl_lines("dataset/response.jsonl"):
        source = sources.get(str(row.get("source_id")), {})
        context = source.get("source_info") or source.get("source")
        prompt = source.get("prompt") or source.get("task_type") or "Assess whether the response is supported by the source."
        response = row.get("response")
        if not context or not response:
            continue
        yield {
            "id": row.get("id"),
            "question": prompt,
            "context": context,
            "response": response,
            "label": response_label(row),
            "source_id": row.get("source_id"),
            "split": row.get("split"),
            "model": row.get("model"),
            "quality": row.get("quality"),
        }


def iter_hf_rows(dataset_payload: Any) -> Iterator[Dict[str, Any]]:
    if hasattr(dataset_payload, "items"):
        for split_name, split in dataset_payload.items():
            for row in split:
                out = dict(row)
                out.setdefault("split_name", split_name)
                yield out
        return
    for row in dataset_payload:
        yield dict(row)


def load_hf_rows(spec: AcquisitionSpec, endpoint: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for automatic Hugging Face acquisition. "
            "Install it or export source JSONL files manually."
        ) from exc
    payload = load_dataset(spec.hf_dataset, revision=spec.revision)  # type: ignore[arg-type]
    yield from iter_hf_rows(payload)


def export_rows(spec: AcquisitionSpec, endpoint: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    if spec.key == "ragtruth":
        return export_ragtruth_rows()
    raw_rows = load_hf_rows(spec, endpoint=endpoint)
    if spec.key == "helpsteer2":
        return export_helpsteer2_rows(raw_rows)
    if spec.key == "oasst1":
        return export_oasst1_pairs(raw_rows)
    if spec.key == "offsetbias":
        return export_offsetbias_rows(raw_rows)
    return raw_rows


def selected_specs(source_names: Sequence[str], include_rewardbench: bool) -> List[AcquisitionSpec]:
    if not source_names or "training" in source_names:
        keys = [key for key, spec in SOURCE_CONFIGS.items() if spec.training_admission]
    elif "all" in source_names:
        keys = list(SOURCE_CONFIGS)
    else:
        keys = list(source_names)
    if include_rewardbench and "rewardbench" not in keys:
        keys.append("rewardbench")
    specs: List[AcquisitionSpec] = []
    for key in keys:
        if key not in SOURCE_CONFIGS:
            raise ValueError(f"unknown source: {key}")
        specs.append(SOURCE_CONFIGS[key])
    return specs


def metadata_for_source(source_dir: Path, spec: AcquisitionSpec, status: str) -> Dict[str, Any]:
    path = source_dir / spec.filename
    exists = path.exists()
    return {
        "url": spec.url,
        "path": path_relative_to_root(path),
        "acquisition_date": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        if exists
        else None,
        "license": spec.license,
        "license_url": spec.license_url,
        "revision": spec.revision,
        "sha256": sha256_file(path),
        "record_count": count_jsonl(path),
        "training_admission": spec.training_admission,
        "external_eval_only": spec.external_eval_only,
        "status": status,
    }


def acquire_sources(
    *,
    source_dir: Path,
    metadata_path: Path,
    sources: Sequence[str],
    include_rewardbench: bool = False,
    metadata_only: bool = False,
    skip_existing: bool = True,
    limit_per_source: Optional[int] = None,
    hf_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    source_dir.mkdir(parents=True, exist_ok=True)
    reports: Dict[str, Any] = {}
    existing = read_existing_metadata(metadata_path)
    metadata_sources: Dict[str, Any] = dict(existing.get("sources", {})) if isinstance(existing.get("sources"), dict) else {}
    for spec in selected_specs(sources, include_rewardbench):
        path = source_dir / spec.filename
        status = "metadata_only"
        if metadata_only:
            status = "metadata_only_existing" if path.exists() else "metadata_only_missing"
        elif skip_existing and path.exists():
            status = "skipped_existing"
        elif spec.hf_dataset or spec.key == "ragtruth":
            rows = export_rows(spec, endpoint=hf_endpoint)
            write_jsonl(path, rows, limit=limit_per_source)
            status = "downloaded"
        else:
            status = "manual_required"

        item = metadata_for_source(source_dir, spec, status)
        reports[spec.key] = item
        metadata_sources[spec.key] = item

    payload = {"created_at": utc_now(), "sources": metadata_sources}
    write_json(metadata_path, payload)
    return {"created_at": payload["created_at"], "reports": reports, "metadata_path": path_relative_to_root(metadata_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Acquire or audit BEA-Judge-10K v2 raw sources.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--sources", nargs="*", default=["training"])
    parser.add_argument("--include-rewardbench", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--hf-endpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = acquire_sources(
        source_dir=args.source_dir,
        metadata_path=args.metadata,
        sources=args.sources,
        include_rewardbench=args.include_rewardbench,
        metadata_only=args.metadata_only,
        skip_existing=not args.overwrite,
        limit_per_source=args.limit_per_source,
        hf_endpoint=args.hf_endpoint,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
