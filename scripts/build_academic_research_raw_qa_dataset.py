"""Build a bilingual academic-research Raw QA dataset.

The script intentionally uses only the Python standard library for network
access so it can run in the project environment without installing extra
packages. It writes a reproducible 20k-ish dataset under datasets/方案2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency fallback.
    pq = None


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "datasets" / "方案2"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "raw_cache"
SEED = 20260609
RNG = random.Random(SEED)

OPEN_LICENSES = {
    "apache-2.0",
    "cc-by-4.0",
    "cc-by",
    "mit",
    "bsd-3-clause",
    "internal/self-built",
}
RESTRICTED_LICENSE_MARKERS = (
    "unknown",
    "non-commercial",
    "cc-by-nc",
    "no redistribution",
    "research only",
)

ACADEMIC_RE = re.compile(
    r"\b("
    r"research|scientific|science|study|experiment|methodology|hypothesis|"
    r"paper|journal|conference|literature review|systematic review|abstract|"
    r"university|professor|student|course|biology|chemistry|physics|medicine|"
    r"medical|psychology|economics|engineering|computer science|statistics|"
    r"mathematics|law|history|philosophy"
    r")\b",
    re.IGNORECASE,
)

SOURCE_LICENSES = {
    "qasper": "CC-BY-4.0",
    "medquad": "CC-BY-4.0",
    "pubmedqa": "MIT",
    "scifact": "Apache-2.0",
    "qasc": "CC-BY",
    "sciq": "CC-BY-4.0",
    "csl_title": "Apache-2.0",
    "csl_keywords": "Apache-2.0",
    "zh_professional": "internal/self-built",
    "ragtruth": "MIT",
    "helpsteer2": "CC-BY-4.0",
    "oasst1": "Apache-2.0",
    "offsetbias": "BSD-3-Clause",
    "pandalm": "Apache-2.0",
    "mt_bench": "CC-BY-4.0",
    "judgebench": "Apache-2.0",
    "synthetic_perturbed": "Apache-2.0",
    "ares_nq": "Apache-2.0",
    "wikieval": "CC-BY-4.0",
}

SOURCE_URLS = {
    "qasper": "https://huggingface.co/datasets/allenai/qasper",
    "medquad": "https://huggingface.co/datasets/lavita/MedQuAD",
    "pubmedqa": "https://github.com/pubmedqa/pubmedqa",
    "scifact": "https://huggingface.co/datasets/bigbio/scifact",
    "qasc": "https://huggingface.co/datasets/allenai/qasc",
    "sciq": "https://huggingface.co/datasets/allenai/sciq",
    "csl_title": "https://github.com/ydli-ai/CSL",
    "csl_keywords": "https://github.com/ydli-ai/CSL",
    "zh_professional": "local:datasets/splits_zh",
}


@dataclass(frozen=True)
class SourceReport:
    source_dataset: str
    language: str
    target_records: int
    built_records: int
    license: str
    license_status: str
    source_url: str
    admission_status: str
    notes: str = ""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any, max_chars: Optional[int] = None) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text.replace("\u0000", " ")).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def stable_hash(*parts: Any, length: int = 16) -> str:
    joined = "\n".join(normalize_text(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def license_status(license_name: str) -> str:
    lowered = license_name.lower()
    if any(marker in lowered for marker in RESTRICTED_LICENSE_MARKERS):
        return "eval_only"
    return "admitted" if lowered in OPEN_LICENSES else "eval_only"


def make_record(
    *,
    source_dataset: str,
    language: str,
    domain: str,
    task_type: str,
    question: Any,
    answer: Any,
    context: Any = "",
    evidence: Any = "",
    options: Optional[Dict[str, str]] = None,
    label: Any = "",
    original_id: Any = "",
    source_url: Optional[str] = None,
    license_name: Optional[str] = None,
    group_key: Any = "",
) -> Optional[Dict[str, Any]]:
    question_text = normalize_text(question, 5000)
    answer_text = normalize_text(answer, 5000)
    if not question_text or not answer_text:
        return None
    source_url = source_url or SOURCE_URLS.get(source_dataset, "")
    license_name = license_name or SOURCE_LICENSES.get(source_dataset, "unknown")
    original_id_text = normalize_text(original_id) or stable_hash(source_dataset, question_text, answer_text)
    record_id = f"{source_dataset}_{stable_hash(language, source_dataset, original_id_text, question_text)}"
    group = normalize_text(group_key) or f"{source_dataset}:{original_id_text}"
    return {
        "id": record_id,
        "language": language,
        "source_dataset": source_dataset,
        "source_url": source_url,
        "license": license_name,
        "license_status": license_status(license_name),
        "domain": domain,
        "task_type": task_type,
        "question": question_text,
        "answer": answer_text,
        "options": options or {},
        "context": normalize_text(context, 12000),
        "evidence": normalize_text(evidence, 8000),
        "label": normalize_text(label),
        "original_id": original_id_text,
        "split": "",
        "group_key": group,
    }


def http_get(url: str, *, timeout: int = 60, retries: int = 4) -> bytes:
    last_error: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "BEA-Judge-academic-raw-qa-builder/1.0",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # pragma: no cover - network retries are environment-dependent.
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}")


def cache_bytes(cache_dir: Path, name: str, url: str) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / name
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    data = http_get(url)
    path.write_bytes(data)
    return data


def cache_text(cache_dir: Path, name: str, url: str) -> str:
    return cache_bytes(cache_dir, name, url).decode("utf-8", errors="replace")


def hf_rows(
    cache_dir: Path,
    *,
    dataset: str,
    config: str,
    split: str,
    page_length: int = 100,
    row_limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    if pq is not None:
        try:
            yield from hf_rows_from_parquet(
                cache_dir,
                dataset=dataset,
                config=config,
                split=split,
                row_limit=row_limit,
            )
            return
        except Exception:
            # Fall back to the rows API if parquet acquisition fails.
            pass

    offset = 0
    yielded = 0
    while True:
        params = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_length,
            }
        )
        url = f"https://datasets-server.huggingface.co/rows?{params}"
        cache_name = f"hf_{safe_name(dataset)}_{safe_name(config)}_{safe_name(split)}_{offset}_{page_length}.json"
        payload = json.loads(cache_bytes(cache_dir, cache_name, url).decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            yield item.get("row", {})
            yielded += 1
            if row_limit is not None and yielded >= row_limit:
                return
        offset += len(rows)
        total = int(payload.get("num_rows_total") or 0)
        if offset >= total:
            break


def hf_rows_from_parquet(
    cache_dir: Path,
    *,
    dataset: str,
    config: str,
    split: str,
    row_limit: Optional[int] = None,
    batch_size: int = 128,
) -> Iterator[Dict[str, Any]]:
    params = urllib.parse.urlencode({"dataset": dataset, "config": config})
    parquet_url = f"https://datasets-server.huggingface.co/parquet?{params}"
    listing_name = f"hf_parquet_listing_{safe_name(dataset)}_{safe_name(config)}.json"
    payload = json.loads(cache_bytes(cache_dir, listing_name, parquet_url).decode("utf-8"))
    parquet_files = payload.get("parquet_files") or []
    matching = [item for item in parquet_files if item.get("split") == split and item.get("config") == config]
    if not matching:
        raise RuntimeError(f"no parquet files for {dataset}/{config}/{split}")

    yielded = 0
    for index, item in enumerate(matching):
        url = item["url"]
        file_name = f"hf_{safe_name(dataset)}_{safe_name(config)}_{safe_name(split)}_{index}.parquet"
        local_path = cache_dir / file_name
        if not local_path.exists() or local_path.stat().st_size == 0:
            local_path.write_bytes(http_get(url, timeout=120))
        parquet_file = pq.ParquetFile(str(local_path))
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                yield row
                yielded += 1
                if row_limit is not None and yielded >= row_limit:
                    return


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def parse_qasper_answer(answer_entry: Any) -> Tuple[str, str, str]:
    answers = []
    if isinstance(answer_entry, dict):
        answers = answer_entry.get("answer") or []
    elif isinstance(answer_entry, list):
        answers = answer_entry
    if isinstance(answers, dict):
        answers = [answers]
    fallback = ("Unanswerable", "", "unanswerable")
    for candidate in answers:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("unanswerable"):
            fallback = ("Unanswerable", "", "unanswerable")
            continue
        yes_no = candidate.get("yes_no")
        free_form = first_nonempty(candidate.get("free_form_answer"))
        extractive = candidate.get("extractive_spans") or []
        if yes_no is not None:
            answer = "Yes" if bool(yes_no) else "No"
        elif free_form:
            answer = free_form
        elif extractive:
            answer = "; ".join(normalize_text(span) for span in extractive if normalize_text(span))
        else:
            answer = ""
        evidence = candidate.get("evidence") or candidate.get("highlighted_evidence") or []
        if answer:
            return answer, " ".join(normalize_text(e) for e in evidence if normalize_text(e)), "answerable"
    return fallback


def collect_qasper(cache_dir: Path, target: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for row in hf_rows(cache_dir, dataset="allenai/qasper", config="qasper", split=split, page_length=20):
            qas = row.get("qas") or {}
            questions = qas.get("question") or []
            question_ids = qas.get("question_id") or []
            answer_entries = qas.get("answers") or []
            context = f"Title: {normalize_text(row.get('title'))}\nAbstract: {normalize_text(row.get('abstract'))}"
            paper_id = normalize_text(row.get("id"))
            for idx, question in enumerate(questions):
                answer, evidence, label = parse_qasper_answer(answer_entries[idx] if idx < len(answer_entries) else {})
                record = make_record(
                    source_dataset="qasper",
                    language="en",
                    domain="scientific_literature",
                    task_type="paper_qa",
                    question=question,
                    answer=answer,
                    context=context,
                    evidence=evidence,
                    label=label,
                    original_id=question_ids[idx] if idx < len(question_ids) else f"{paper_id}_{idx}",
                    group_key=f"qasper:{paper_id}",
                )
                if record:
                    records.append(record)
                    if len(records) >= target:
                        return records
    return records


def collect_medquad(cache_dir: Path, target: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for row in hf_rows(cache_dir, dataset="lavita/MedQuAD", config="default", split="train", page_length=100):
        context_parts = [
            f"Document source: {normalize_text(row.get('document_source'))}",
            f"Question focus: {normalize_text(row.get('question_focus'))}",
            f"Question type: {normalize_text(row.get('question_type'))}",
            f"Document URL: {normalize_text(row.get('document_url'))}",
        ]
        record = make_record(
            source_dataset="medquad",
            language="en",
            domain="biomedical",
            task_type="qa",
            question=row.get("question"),
            answer=row.get("answer"),
            context="\n".join(context_parts),
            evidence=row.get("answer"),
            label=row.get("question_type"),
            original_id=row.get("question_id") or row.get("document_id"),
            source_url=SOURCE_URLS["medquad"],
            group_key=f"medquad:{row.get('document_id')}",
        )
        if record:
            records.append(record)
            if len(records) >= target:
                break
    return records


def collect_pubmedqa(cache_dir: Path, target: int) -> List[Dict[str, Any]]:
    url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    data = json.loads(cache_text(cache_dir, "pubmedqa_ori_pqal.json", url))
    records: List[Dict[str, Any]] = []
    for pmid, row in data.items():
        contexts = row.get("CONTEXTS") or row.get("contexts") or []
        context = " ".join(normalize_text(item) for item in contexts if normalize_text(item))
        answer = first_nonempty(row.get("LONG_ANSWER"), row.get("long_answer"), row.get("final_decision"))
        record = make_record(
            source_dataset="pubmedqa",
            language="en",
            domain="biomedical",
            task_type="qa",
            question=row.get("QUESTION") or row.get("question"),
            answer=answer,
            context=context,
            evidence=context,
            label=row.get("final_decision", ""),
            original_id=pmid,
            source_url=SOURCE_URLS["pubmedqa"],
            group_key=f"pubmedqa:{pmid}",
        )
        if record:
            records.append(record)
            if len(records) >= target:
                break
    return records


def collect_scifact(cache_dir: Path, target: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for split in ("train", "validation"):
        for row in hf_rows(
            cache_dir,
            dataset="bigbio/scifact",
            config="scifact_labelprediction_bigbio_pairs",
            split=split,
            page_length=100,
        ):
            label = normalize_text(row.get("label"))
            answer = {
                "SUPPORT": "The evidence supports the claim.",
                "CONTRADICT": "The evidence contradicts the claim.",
                "NOINFO": "The evidence does not provide enough information for the claim.",
            }.get(label, f"Evidence label: {label}")
            record = make_record(
                source_dataset="scifact",
                language="en",
                domain="scientific_claim_verification",
                task_type="claim_verification",
                question=f"Does the following evidence support the claim? Claim: {normalize_text(row.get('text_1'))}",
                answer=answer,
                context=row.get("text_2"),
                evidence=row.get("text_2"),
                label=label,
                original_id=row.get("id"),
                source_url=SOURCE_URLS["scifact"],
                group_key=f"scifact:{row.get('id')}",
            )
            if record:
                records.append(record)
                if len(records) >= target:
                    return records
    return records


def collect_qasc(cache_dir: Path, target: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for row in hf_rows(cache_dir, dataset="allenai/qasc", config="default", split=split, page_length=100):
            choices = row.get("choices") or {}
            labels = choices.get("label") or []
            texts = choices.get("text") or []
            options = {normalize_text(label): normalize_text(text) for label, text in zip(labels, texts)}
            answer_key = normalize_text(row.get("answerKey"))
            answer = options.get(answer_key, answer_key)
            context = "\n".join(
                part
                for part in [
                    normalize_text(row.get("fact1")),
                    normalize_text(row.get("fact2")),
                    normalize_text(row.get("combinedfact")),
                ]
                if part
            )
            record = make_record(
                source_dataset="qasc",
                language="en",
                domain="stem",
                task_type="multi_choice",
                question=row.get("question") or row.get("formatted_question"),
                answer=answer,
                options=options,
                context=context,
                evidence=context,
                label=answer_key,
                original_id=row.get("id"),
                source_url=SOURCE_URLS["qasc"],
                group_key=f"qasc:{row.get('id')}",
            )
            if record:
                records.append(record)
                if len(records) >= target:
                    return records
    return records


def collect_sciq(cache_dir: Path, target: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for row in hf_rows(cache_dir, dataset="allenai/sciq", config="default", split=split, page_length=100):
            options = {
                "A": normalize_text(row.get("correct_answer")),
                "B": normalize_text(row.get("distractor1")),
                "C": normalize_text(row.get("distractor2")),
                "D": normalize_text(row.get("distractor3")),
            }
            record = make_record(
                source_dataset="sciq",
                language="en",
                domain="stem",
                task_type="multi_choice",
                question=row.get("question"),
                answer=row.get("correct_answer"),
                options=options,
                context=row.get("support"),
                evidence=row.get("support"),
                label="A",
                original_id=f"{split}_{len(records)}",
                source_url=SOURCE_URLS["sciq"],
                group_key=f"sciq:{split}_{len(records)}",
            )
            if record:
                records.append(record)
                if len(records) >= target:
                    return records
    return records


def collect_csl(cache_dir: Path, *, task: str, target: int) -> List[Dict[str, Any]]:
    if task == "title":
        source_dataset = "csl_title"
        url_prefix = "https://raw.githubusercontent.com/ydli-ai/CSL/master/benchmark/ts"
        question = "请根据以下中文学术论文摘要生成论文标题。"
        task_label = "title_generation"
    elif task == "keywords":
        source_dataset = "csl_keywords"
        url_prefix = "https://raw.githubusercontent.com/ydli-ai/CSL/master/benchmark/kg"
        question = "请根据以下中文学术论文内容生成论文关键词。"
        task_label = "keyword_generation"
    else:
        raise ValueError(f"unknown CSL task: {task}")
    records: List[Dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        text = cache_text(cache_dir, f"csl_{task}_{split}.tsv", f"{url_prefix}/{split}.tsv")
        reader = csv.reader(text.splitlines(), delimiter="\t")
        for idx, row in enumerate(reader):
            if len(row) < 3:
                continue
            prompt, text_a, text_b = row[0], row[1], row[2]
            if idx == 0 and prompt in {"prompt", "to title", "to keywords"} and not normalize_text(text_b):
                continue
            record = make_record(
                source_dataset=source_dataset,
                language="zh",
                domain="scientific_literature",
                task_type="metadata_qa",
                question=question,
                answer=text_b,
                context=text_a,
                evidence=text_a,
                label=task_label,
                original_id=f"{split}_{idx}",
                source_url=SOURCE_URLS[source_dataset],
                group_key=f"{source_dataset}:{split}_{idx}",
            )
            if record:
                records.append(record)
                if len(records) >= target:
                    return records
    return records


def load_wrapped_samples(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict):
        for key in ("samples", "records", "data", "items"):
            if isinstance(obj.get(key), list):
                return obj[key]
    return obj if isinstance(obj, list) else []


def is_academic_local_sample(sample: Dict[str, Any]) -> bool:
    metadata = sample.get("metadata") or {}
    if metadata.get("domain") == "科研方法":
        return True
    source_task = normalize_text(metadata.get("source_task"))
    if source_task.startswith(("mmlu-pro-", "livebench-")) or source_task == "livecodebench":
        return True
    text = "\n".join(
        normalize_text(sample.get(key))
        for key in ("prompt", "answer_a", "answer_b", "context", "reference")
    )
    return bool(ACADEMIC_RE.search(text))


def convert_local_sample(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    language = sample.get("language") or ("zh" if sample.get("dataset") == "zh_professional" else "en")
    if language == "zh" and (sample.get("metadata") or {}).get("domain") != "科研方法":
        return None
    dataset = normalize_text(sample.get("dataset"))
    source_dataset = "zh_professional" if dataset == "zh_professional" else dataset
    metadata = sample.get("metadata") or {}
    human_label = normalize_text(sample.get("human_label"))
    answer_a = normalize_text(sample.get("answer_a"))
    answer_b = normalize_text(sample.get("answer_b"))
    options: Dict[str, str] = {}
    answer = answer_a
    task_type = sample.get("task_type")
    if answer_b and answer_b != "[SINGLE_ANSWER_FACTUALITY_TASK]":
        options = {"A": answer_a, "B": answer_b}
        if human_label == "B>A":
            answer = answer_b
        elif human_label == "Tie":
            answer = f"Tie: {answer_a}"
    raw_domain = metadata.get("domain")
    if raw_domain == "科研方法":
        domain = "methodology"
    elif task_type == "factuality_rag":
        domain = "scientific_fact_verification"
    elif dataset in {"judgebench", "synthetic_perturbed"}:
        domain = "academic_reasoning"
    else:
        domain = "academic_open_qa"
    record = make_record(
        source_dataset=source_dataset,
        language=language,
        domain=domain,
        task_type="qa" if task_type == "open_qa" else "claim_verification" if task_type == "factuality_rag" else "multi_choice",
        question=sample.get("prompt"),
        answer=answer,
        options=options,
        context=sample.get("context"),
        evidence=sample.get("reference") or sample.get("context"),
        label=human_label,
        original_id=sample.get("id"),
        source_url=metadata.get("source_url") or SOURCE_URLS.get(source_dataset, ""),
        license_name=SOURCE_LICENSES.get(source_dataset, SOURCE_LICENSES.get(dataset, "CC-BY-4.0")),
        group_key=f"local:{sample.get('id')}",
    )
    if record:
        record["source_dataset"] = f"bea_judge_{source_dataset}"
    return record


def collect_local_academic(root: Path, *, language: str, target: int) -> List[Dict[str, Any]]:
    split_dir = root / "datasets" / ("splits_zh" if language == "zh" else "splits")
    records: List[Dict[str, Any]] = []
    for split_name in ("train", "dev", "test"):
        path = split_dir / f"{split_name}.json"
        if not path.exists():
            continue
        for sample in load_wrapped_samples(path):
            if sample.get("language") != language:
                continue
            if not is_academic_local_sample(sample):
                continue
            record = convert_local_sample(sample)
            if record:
                records.append(record)
                if len(records) >= target:
                    return records
    return records


def dedupe_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []
    for record in records:
        key = stable_hash(
            record.get("language"),
            normalize_text(record.get("question")).lower(),
            normalize_text(record.get("answer")).lower(),
            normalize_text(record.get("context")).lower()[:500],
            length=32,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(record)
    return output


def assign_splits(records: List[Dict[str, Any]]) -> None:
    for record in records:
        value = int(stable_hash(record.get("source_dataset"), record.get("group_key"), length=8), 16) % 1000
        if value < 700:
            record["split"] = "train"
        elif value < 850:
            record["split"] = "dev"
        else:
            record["split"] = "test"


def trim_to_language_quotas(records: List[Dict[str, Any]], *, en_quota: int, zh_quota: int) -> List[Dict[str, Any]]:
    by_language: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_language[record["language"]].append(record)
    selected = by_language["en"][:en_quota] + by_language["zh"][:zh_quota]
    return selected


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def validation_report(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    required = ["id", "language", "source_dataset", "question", "answer", "split", "license"]
    missing = [
        {"id": record.get("id"), "field": field}
        for record in records
        for field in required
        if not normalize_text(record.get(field))
    ]
    id_counts = Counter(record["id"] for record in records)
    duplicate_ids = [record_id for record_id, count in id_counts.items() if count > 1]
    bad_licenses = [
        {"id": record["id"], "license": record.get("license")}
        for record in records
        if any(marker in normalize_text(record.get("license")).lower() for marker in RESTRICTED_LICENSE_MARKERS)
    ]
    by_language = Counter(record["language"] for record in records)
    total_ok = 18000 <= len(records) <= 22000
    en_ok = 11500 <= by_language.get("en", 0) <= 12500
    zh_ok = 7500 <= by_language.get("zh", 0) <= 8500
    schema_ok = not missing and not duplicate_ids
    license_ok = not bad_licenses
    return {
        "total_records": len(records),
        "by_language": dict(by_language),
        "by_split": dict(Counter(record["split"] for record in records)),
        "by_task_type": dict(Counter(record["task_type"] for record in records)),
        "by_domain": dict(Counter(record["domain"] for record in records)),
        "by_source_dataset": dict(Counter(record["source_dataset"] for record in records)),
        "missing_required_fields": missing[:50],
        "duplicate_ids": duplicate_ids[:50],
        "bad_licenses": bad_licenses[:50],
        "checks": {
            "total_18k_22k": total_ok,
            "english_11p5k_12p5k": en_ok,
            "chinese_7p5k_8p5k": zh_ok,
            "schema_required_fields": schema_ok,
            "license_no_restricted_markers": license_ok,
            "all_passed": total_ok and en_ok and zh_ok and schema_ok and license_ok,
        },
    }


def source_reports(records: List[Dict[str, Any]], planned_targets: Dict[str, Tuple[str, int]]) -> List[SourceReport]:
    counts = Counter(record["source_dataset"] for record in records)
    reports: List[SourceReport] = []
    for source, (language, target) in planned_targets.items():
        if source == "zh_professional":
            built_records = sum(
                1 for record in records if record["language"] == "zh" and record["source_dataset"].startswith("bea_judge_zh_professional")
            )
            license_name = "internal/self-built"
            status = "admitted"
            source_url = "local:datasets/splits_zh"
            notes = "Chinese methodology-oriented local supplement extracted from the existing zh_professional split set."
        elif source == "bea_judge_local_en_academic":
            built_records = sum(
                1 for record in records if record["language"] == "en" and record["source_dataset"].startswith("bea_judge_")
            )
            license_name = "mixed_open_inherited"
            status = "admitted"
            source_url = "local:datasets/splits"
            notes = "Aggregate of local BEA-Judge academic English supplements; individual sample licenses are inherited from their source datasets."
        else:
            built_records = counts.get(source, 0)
            license_name = SOURCE_LICENSES.get(source.replace("bea_judge_", ""), SOURCE_LICENSES.get(source, "unknown"))
            status = license_status(license_name)
            source_url = SOURCE_URLS.get(source.replace("bea_judge_", ""), SOURCE_URLS.get(source, ""))
            notes = ""
        reports.append(
            SourceReport(
                source_dataset=source,
                language=language,
                target_records=target,
                built_records=built_records,
                license=license_name,
                license_status=status,
                source_url=source_url,
                admission_status="admitted",
                notes=notes,
            )
        )
    reports.extend(
        [
            SourceReport(
                source_dataset="gpqa",
                language="en",
                target_records=448,
                built_records=0,
                license="MIT",
                license_status="admitted",
                source_url="https://github.com/idavidrein/gpqa",
                admission_status="not_collected",
                notes="HuggingFace dataset access returned 401 in this environment; SciQ was used as an open fallback.",
            ),
            SourceReport(
                source_dataset="mlec_qa",
                language="zh",
                target_records=4500,
                built_records=0,
                license="MIT",
                license_status="admitted",
                source_url="https://github.com/Judenpech/MLEC-QA",
                admission_status="not_collected",
                notes="Repository is MIT, but dataset payload is distributed via Google Drive; direct scripted acquisition was not used.",
            ),
            SourceReport(
                source_dataset="ceval_exam",
                language="zh",
                target_records=0,
                built_records=0,
                license="CC-BY-NC-SA-4.0",
                license_status="eval_only",
                source_url="https://huggingface.co/datasets/ceval/ceval-exam",
                admission_status="eval_only",
                notes="Non-commercial license; excluded from admitted training dataset.",
            ),
        ]
    )
    return reports


def write_statistics(output_dir: Path, records: List[Dict[str, Any]], reports: List[SourceReport]) -> None:
    stats = validation_report(records)
    write_json(output_dir / "dataset_statistics.json", stats)
    lines = [
        "# Academic Research Raw QA Dataset Statistics",
        "",
        f"- Created at: {utc_now()}",
        f"- Total records: {stats['total_records']}",
        f"- Language distribution: {stats['by_language']}",
        f"- Split distribution: {stats['by_split']}",
        f"- Validation passed: {stats['checks']['all_passed']}",
        "",
        "## By Source Dataset",
        "",
        "| source_dataset | records |",
        "| --- | ---: |",
    ]
    for source, count in Counter(record["source_dataset"] for record in records).most_common():
        lines.append(f"| {source} | {count} |")
    lines.extend(["", "## By Domain", "", "| domain | records |", "| --- | ---: |"])
    for domain, count in Counter(record["domain"] for record in records).most_common():
        lines.append(f"| {domain} | {count} |")
    (output_dir / "dataset_statistics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = [
        "| source_dataset | language | target | built | license | license_status | admission_status | source_url | notes |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        provenance.append(
            "| {source_dataset} | {language} | {target_records} | {built_records} | {license} | {license_status} | {admission_status} | {source_url} | {notes} |".format(
                **report.__dict__
            )
        )
    (output_dir / "source_provenance_table.md").write_text("\n".join(provenance) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path, records: List[Dict[str, Any]], reports: List[SourceReport]) -> None:
    manifest = {
        "created_at": utc_now(),
        "objective": "Bilingual academic-research Raw QA dataset, English-heavy 60/40 target.",
        "target_records": 20000,
        "built_records": len(records),
        "seed": SEED,
        "schema": {
            "id": "stable sample id",
            "language": "en|zh",
            "source_dataset": "source identifier",
            "source_url": "source homepage or local path",
            "license": "source license",
            "license_status": "admitted|eval_only|excluded",
            "domain": "academic-research domain",
            "task_type": "qa|multi_choice|claim_verification|paper_qa|metadata_qa",
            "question": "raw question or deterministic prompt",
            "answer": "gold/reference answer",
            "options": "multiple-choice options when applicable",
            "context": "source context",
            "evidence": "supporting evidence when available",
            "label": "source label or answer key",
            "original_id": "source sample id",
            "split": "train|dev|test",
        },
        "files": {
            "jsonl": "academic_research_raw_qa.jsonl",
            "json": "academic_research_raw_qa.json",
            "splits": "splits/{train,dev,test}.json",
            "statistics": "dataset_statistics.md",
            "provenance": "source_provenance_table.md",
        },
        "sources": [report.__dict__ for report in reports],
        "validation": validation_report(records),
    }
    write_json(output_dir / "data_manifest.json", manifest)


def build_dataset(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[SourceReport]]:
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "raw_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    collectors = [
        ("qasper", "en", 4500, lambda: collect_qasper(cache_dir, 4500)),
        ("medquad", "en", 2000, lambda: collect_medquad(cache_dir, 2000)),
        ("pubmedqa", "en", 1000, lambda: collect_pubmedqa(cache_dir, 1000)),
        ("scifact", "en", 1000, lambda: collect_scifact(cache_dir, 1000)),
        ("qasc", "en", 1000, lambda: collect_qasc(cache_dir, 1000)),
        ("sciq", "en", 448, lambda: collect_sciq(cache_dir, 448)),
        ("csl_title", "zh", 4020, lambda: collect_csl(cache_dir, task="title", target=4020)),
        ("csl_keywords", "zh", 3760, lambda: collect_csl(cache_dir, task="keywords", target=3760)),
        ("zh_professional", "zh", 276, lambda: collect_local_academic(ROOT, language="zh", target=276)),
    ]

    planned_targets = {name: (language, target) for name, language, target, _ in collectors}
    all_records: List[Dict[str, Any]] = []
    for name, language, target, collector in collectors:
        print(f"[collect] {name} target={target} language={language}", flush=True)
        records = collector()
        print(f"[collect] {name} built={len(records)}", flush=True)
        all_records.extend(records)

    print("[collect] bea_judge English academic subset target=2600 language=en", flush=True)
    local_en = collect_local_academic(ROOT, language="en", target=2600)
    print(f"[collect] bea_judge English academic subset built={len(local_en)}", flush=True)
    all_records.extend(local_en)
    planned_targets["bea_judge_local_en_academic"] = ("en", 2600)

    all_records = dedupe_records(all_records)
    all_records = trim_to_language_quotas(all_records, en_quota=args.en_quota, zh_quota=args.zh_quota)
    assign_splits(all_records)
    reports = source_reports(all_records, planned_targets)
    return all_records, reports


def write_outputs(output_dir: Path, records: List[Dict[str, Any]], reports: List[SourceReport]) -> None:
    write_jsonl(output_dir / "academic_research_raw_qa.jsonl", records)
    write_json(output_dir / "academic_research_raw_qa.json", {"metadata": {"created_at": utc_now()}, "samples": records})
    split_dir = output_dir / "splits"
    for split in ("train", "dev", "test"):
        split_records = [record for record in records if record["split"] == split]
        write_json(split_dir / f"{split}.json", {"metadata": {"split": split, "count": len(split_records)}, "samples": split_records})
    write_statistics(output_dir, records, reports)
    write_manifest(output_dir, records, reports)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--en-quota", type=int, default=12000)
    parser.add_argument("--zh-quota", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, reports = build_dataset(args)
    write_outputs(Path(args.output_dir), records, reports)
    report = validation_report(records)
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2), flush=True)
    if not report["checks"]["all_passed"]:
        raise SystemExit("dataset validation failed")


if __name__ == "__main__":
    main()
