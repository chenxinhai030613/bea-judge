"""
BEA-Judge core dataset builder.

This script builds the three public-data core sets requested by the
BEA-Judge experimental design:

1. Open-ended answer quality: MT-Bench human judgments + PandaLM test set.
2. Judge bias and difficult pairs: JudgeBench + deterministic perturbations.
3. Factuality and RAG: RAGAS WikiEval + ARES Natural Questions labels.

Outputs are normalized to the BEA-Judge JSON schema and include deterministic
train/dev/test splits, source metadata, quality-control statistics, and raw
download manifests.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SEED = 42
RNG = random.Random(SEED)
SPLIT_RATIOS = (0.70, 0.15, 0.15)
DEFAULT_TARGET_PER_TASK = 400
MIN_TEXT_CHARS = 10

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "datasets"
RAW_DIR = OUTPUT_DIR / "raw"
PROCESSED_DIR = OUTPUT_DIR / "processed"
SPLIT_DIR = OUTPUT_DIR / "splits"
DEPS_DIR = PROJECT_ROOT / "_deps"

if DEPS_DIR.exists():
    # Only prepend bundled deps when compiled wheels match the active Python.
    pyarrow_dir = DEPS_DIR / "pyarrow"
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    pyarrow_compatible = (
        not pyarrow_dir.exists()
        or any(pyarrow_dir.glob(f"lib.{py_tag}*.pyd"))
        or any(pyarrow_dir.glob(f"_*.{py_tag}*.pyd"))
    )
    if pyarrow_compatible:
        sys.path.insert(0, str(DEPS_DIR))

SOURCE_URLS = {
    "mt_bench_human": "https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/main/data/human-00000-of-00001-25f4910818759289.parquet",
    "pandalm_test_api": "https://api.github.com/repos/WeOpenML/PandaLM/contents/data/testset-v1.json?ref=main",
    "judgebench_claude": "https://huggingface.co/datasets/ScalerLab/JudgeBench/resolve/main/data/claude-00000-of-00001.jsonl",
    "wikieval": "https://huggingface.co/datasets/vibrantlabsai/WikiEval/resolve/main/data/train-00000-of-00001-385c01e94624e9b7.parquet",
    "ares_nq_labeled_api_raw": "https://api.github.com/repos/stanford-futuredata/ARES/contents/datasets/example_files/nq_labeled_output.tsv?ref=main",
}

CORE_FIELDS = [
    "id",
    "dataset",
    "task_type",
    "prompt",
    "context",
    "answer_a",
    "answer_b",
    "reference",
    "human_score",
    "human_label",
]

OPTIONAL_TEXT_FIELDS = ("context", "answer_b", "reference")
TASK_FIELD_CONTRACTS = {
    "open_qa": {
        "context": "optional",
        "answer_b": "required",
        "reference": "optional",
    },
    "pairwise_bias": {
        "context": "optional",
        "answer_b": "required",
        "reference": "optional",
    },
    "factuality_rag": {
        "context": "required",
        "answer_b": "required",
        "reference": "recommended",
    },
}
PAIRWISE_LABEL_VALUES = {"A>B": 1.0, "B>A": -1.0, "Tie": 0.0}
FACTUALITY_LABEL_VALUES = {"supported": 1.0, "ambiguous": 0.5, "unsupported": 0.0}
VALID_HUMAN_LABELS = set(PAIRWISE_LABEL_VALUES) | set(FACTUALITY_LABEL_VALUES) | {None}
PHONE_LIKE_RE = re.compile(r"\b\d{3}[-.]?\d{3,4}[-.]?\d{4}\b")
PHONE_DASH_DOT_RE = re.compile(r"\b\d{3}[-.]\d{3,4}[-.]\d{4}\b")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, RAW_DIR, PROCESSED_DIR, SPLIT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_text(value: Any, max_chars: Optional[int] = None) -> str:
    if value is None:
        text = ""
    elif isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    elif hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            text = "\n\n".join(normalize_text(x) for x in converted)
        else:
            text = str(converted)
    elif isinstance(value, (list, tuple)):
        text = "\n\n".join(normalize_text(x) for x in value)
    else:
        text = str(value)

    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + " ... [TRUNCATED]"
    return text


def scrub_phone_like_text(text: str) -> Tuple[str, List[Dict[str, str]]]:
    findings: List[Dict[str, str]] = []

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        if PHONE_DASH_DOT_RE.fullmatch(value):
            findings.append({"match": value, "action": "redacted_phone_like_token"})
            return "[PHONE_LIKE_TOKEN]"
        findings.append({"match": value, "action": "numeric_false_positive_retained"})
        return value

    return PHONE_LIKE_RE.sub(replace, text), findings


def strip_prefix(text: str, prefixes: Iterable[str]) -> str:
    out = text.strip()
    for prefix in prefixes:
        if out.lower().startswith(prefix.lower()):
            return out[len(prefix) :].strip()
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: str) -> str:
    joined = "\u241f".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def download_file(
    url: str,
    output_path: Path,
    *,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 4,
    timeout: int = 180,
) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    headers = {"User-Agent": "Mozilla/5.0", **(headers or {})}
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with output_path.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            return output_path
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if output_path.exists():
                output_path.unlink()
            time.sleep(min(2 * attempt, 8))
    raise RuntimeError(f"Failed to download {url}: {last_error}")


def download_github_api_base64(url: str, output_path: Path) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    content = payload.get("content")
    if not content:
        raise RuntimeError(f"GitHub API response does not contain base64 content: {url}")
    decoded = base64.b64decode(content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(decoded)
    return output_path


def read_parquet(path: Path) -> List[Dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    except ImportError:
        pass
    except Exception:
        # Fall back to pyarrow path when pandas exists but cannot deserialize
        # the source due to optional engine/runtime differences.
        pass

    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Parquet reader unavailable: install pandas or pyarrow into "
            f"{DEPS_DIR} (or current Python environment)."
        ) from exc

    table = pq.read_table(path)
    rows = table.to_pylist()
    return [row if isinstance(row, dict) else dict(row) for row in rows]


def label_from_winner(winner: Any) -> str:
    value = normalize_text(winner).lower()
    if value in {"model_a", "a", "answer_a", "response_a"}:
        return "A>B"
    if value in {"model_b", "b", "answer_b", "response_b"}:
        return "B>A"
    return "Tie"


def pairwise_label_score(label: Optional[str]) -> Optional[float]:
    if label is None:
        return None
    return PAIRWISE_LABEL_VALUES.get(label)


def factuality_label_score(label: Optional[str]) -> Optional[float]:
    if label is None:
        return None
    return FACTUALITY_LABEL_VALUES.get(label)


def invert_label(label: Optional[str]) -> Optional[str]:
    if label == "A>B":
        return "B>A"
    if label == "B>A":
        return "A>B"
    return label


def majority_pandalm_label(row: Dict[str, Any]) -> str:
    votes = []
    for key in ("annotator1", "annotator2", "annotator3"):
        value = row.get(key)
        if value == 1:
            votes.append("A>B")
        elif value == 2:
            votes.append("B>A")
        else:
            votes.append("Tie")
    counts = Counter(votes)
    label, count = counts.most_common(1)[0]
    if count < 2:
        return "Tie"
    return label


def conversation_parts(conversation: Any) -> Tuple[str, str, str]:
    messages: List[Dict[str, Any]] = []
    if hasattr(conversation, "tolist"):
        conversation = conversation.tolist()
    for message in conversation or []:
        if isinstance(message, dict):
            messages.append(message)

    user_messages = [normalize_text(m.get("content")) for m in messages if m.get("role") == "user"]
    assistant_messages = [
        normalize_text(m.get("content"), max_chars=4000)
        for m in messages
        if m.get("role") == "assistant"
    ]

    prompt = "\n".join(f"Turn {idx + 1}: {text}" for idx, text in enumerate(user_messages))
    answer = assistant_messages[-1] if assistant_messages else ""

    context_chunks = []
    assistant_count = len(assistant_messages)
    if len(user_messages) > 1 and assistant_count > 1:
        for idx in range(min(len(user_messages) - 1, assistant_count - 1)):
            context_chunks.append(f"User: {user_messages[idx]}")
            context_chunks.append(f"Assistant: {assistant_messages[idx]}")
    context = "\n".join(context_chunks)
    return prompt, context, answer


def detect_language(*texts: str) -> str:
    joined = "".join(texts[:2])
    zh_chars = sum(1 for char in joined if "\u4e00" <= char <= "\u9fff")
    return "zh" if zh_chars >= 5 else "en"


def base_sample(
    *,
    sample_id: str,
    dataset: str,
    task_type: str,
    prompt: str,
    answer_a: str,
    context: str = "",
    answer_b: str = "",
    reference: str = "",
    human_score: Optional[Dict[str, Any]] = None,
    human_label: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    prompt = normalize_text(prompt, max_chars=4000)
    context = normalize_text(context, max_chars=8000)
    answer_a = normalize_text(answer_a, max_chars=5000)
    answer_b = normalize_text(answer_b, max_chars=5000)
    reference = normalize_text(reference, max_chars=4000)
    metadata = metadata or {}
    pii_findings: List[Dict[str, str]] = []
    for field_name, field_value in (
        ("prompt", prompt),
        ("context", context),
        ("answer_a", answer_a),
        ("answer_b", answer_b),
        ("reference", reference),
    ):
        scrubbed, findings = scrub_phone_like_text(field_value)
        if findings:
            pii_findings.extend({"field": field_name, **finding} for finding in findings)
        if field_name == "prompt":
            prompt = scrubbed
        elif field_name == "context":
            context = scrubbed
        elif field_name == "answer_a":
            answer_a = scrubbed
        elif field_name == "answer_b":
            answer_b = scrubbed
        else:
            reference = scrubbed

    if pii_findings:
        metadata["pii_review"] = {
            "status": "reviewed",
            "findings": pii_findings,
        }
    score_format = normalize_text(metadata.get("score_format") or "unknown")
    scoring_system = normalize_text(metadata.get("scoring_system") or score_format or "unknown")
    human_score = human_score or {}
    if "score_format" not in human_score:
        human_score["score_format"] = score_format
    if "scoring_system" not in human_score:
        human_score["scoring_system"] = scoring_system
    if human_label is not None and "label" not in human_score:
        human_score["label"] = human_label
    metadata.update(
        {
            "score_format": score_format,
            "scoring_system": scoring_system,
            "prompt_chars": len(prompt),
            "context_chars": len(context),
            "answer_a_chars": len(answer_a),
            "answer_b_chars": len(answer_b),
        }
    )

    return {
        "id": sample_id,
        "dataset": dataset,
        "task_type": task_type,
        "prompt": prompt,
        "context": context,
        "answer_a": answer_a,
        "answer_b": answer_b,
        "reference": reference,
        "human_score": human_score,
        "human_label": human_label,
        "language": detect_language(prompt, answer_a),
        "metadata": metadata,
    }


def valid_sample(sample: Dict[str, Any], *, require_pair: bool = False) -> Tuple[bool, str]:
    for field in ("id", "dataset", "task_type", "prompt", "answer_a"):
        if not sample.get(field):
            return False, f"missing_{field}"
    if len(sample["prompt"]) < 8:
        return False, "short_prompt"
    if len(sample["answer_a"]) < MIN_TEXT_CHARS:
        return False, "short_answer_a"
    if require_pair and len(sample.get("answer_b", "")) < MIN_TEXT_CHARS:
        return False, "short_answer_b"
    if sample.get("human_label") not in VALID_HUMAN_LABELS:
        return False, "bad_human_label"
    if sample.get("task_type") == "factuality_rag" and not sample.get("context"):
        return False, "missing_context"
    return True, "ok"


def deduplicate(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    seen = set()
    kept = []
    removed = 0
    removed_entries: List[Dict[str, Any]] = []
    for sample in samples:
        key = stable_hash(
            sample.get("task_type", ""),
            sample.get("prompt", ""),
            sample.get("answer_a", ""),
            sample.get("answer_b", ""),
        )
        if key in seen:
            removed += 1
            removed_entries.append(
                {
                    "reason": "duplicate",
                    "task_type": sample.get("task_type"),
                    "dataset": sample.get("dataset"),
                    "id": sample.get("id"),
                }
            )
            continue
        seen.add(key)
        kept.append(sample)
    return kept, removed, removed_entries


def balanced_select(
    samples: List[Dict[str, Any]],
    target: int,
    label_getter=lambda sample: sample.get("human_label") or "NA",
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(label_getter(sample))].append(sample)
    for group in groups.values():
        RNG.shuffle(group)

    selected: List[Dict[str, Any]] = []
    labels = sorted(groups)
    while len(selected) < target and labels:
        progressed = False
        for label in list(labels):
            if groups[label] and len(selected) < target:
                selected.append(groups[label].pop())
                progressed = True
            if not groups[label]:
                labels.remove(label)
        if not progressed:
            break
    return selected[:target]


def build_open_qa(mt_path: Path, pandalm_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    mt_rows = read_parquet(mt_path)
    mt_records: List[Dict[str, Any]] = []
    for row in mt_rows:
        prompt_a, context_a, answer_a = conversation_parts(row.get("conversation_a"))
        _, _, answer_b = conversation_parts(row.get("conversation_b"))
        label = label_from_winner(row.get("winner"))
        sample = base_sample(
            sample_id="pending",
            dataset="mt_bench",
            task_type="open_qa",
            prompt=prompt_a,
            context=context_a,
            answer_a=answer_a,
            answer_b=answer_b,
            human_label=label,
            human_score={
                "score_format": "pairwise_preference",
                "scoring_system": "pairwise_preference",
                "pairwise_preference": pairwise_label_score(label),
                "label": label,
            },
            metadata={
                "source": "lmsys/mt_bench_human_judgments",
                "source_url": SOURCE_URLS["mt_bench_human"],
                "score_format": "pairwise_preference",
                "scoring_system": "pairwise_preference",
                "question_id": normalize_text(row.get("question_id")),
                "model_a": normalize_text(row.get("model_a")),
                "model_b": normalize_text(row.get("model_b")),
                "winner": normalize_text(row.get("winner")),
                "judge": normalize_text(row.get("judge")),
                "turn": normalize_text(row.get("turn")),
            },
        )
        ok, _ = valid_sample(sample, require_pair=True)
        if ok:
            mt_records.append(sample)

    pandalm_data = json.loads(pandalm_path.read_text(encoding="utf-8"))
    pandalm_records: List[Dict[str, Any]] = []
    for row in pandalm_data:
        instruction = normalize_text(row.get("instruction"))
        input_text = normalize_text(row.get("input"))
        prompt = instruction if not input_text else f"{instruction}\n\nInput: {input_text}"
        label = majority_pandalm_label(row)
        sample = base_sample(
            sample_id="pending",
            dataset="pandalm",
            task_type="open_qa",
            prompt=prompt,
            answer_a=row.get("response1", ""),
            answer_b=row.get("response2", ""),
            human_label=label,
            human_score={
                "score_format": "pairwise_votes",
                "scoring_system": "pairwise_preference",
                "pairwise_preference": pairwise_label_score(label),
                "label": label,
                "annotator_votes": {
                    "annotator1": row.get("annotator1"),
                    "annotator2": row.get("annotator2"),
                    "annotator3": row.get("annotator3"),
                },
            },
            metadata={
                "source": "WeOpenML/PandaLM testset-v1",
                "source_url": SOURCE_URLS["pandalm_test_api"],
                "score_format": "pairwise_votes",
                "scoring_system": "pairwise_preference",
                "original_idx": normalize_text(row.get("idx")),
                "motivation_app": normalize_text(row.get("motivation_app")),
                "cmp_key": normalize_text(row.get("cmp_key")),
                "label_mapping": "1=response1 better, 2=response2 better, otherwise tie",
            },
        )
        ok, _ = valid_sample(sample, require_pair=True)
        if ok:
            pandalm_records.append(sample)

    # Keep the full validated pool, then perform QC and balanced downsampling in
    # the final assembly stage so large-scale expansion can reuse the same code.
    selected = mt_records + pandalm_records
    RNG.shuffle(selected)
    for idx, sample in enumerate(selected, 1):
        sample["id"] = f"open_qa_{idx:04d}"
    return selected


def build_judge_bias(judgebench_path: Path, *, target: int) -> List[Dict[str, Any]]:
    rows = []
    with judgebench_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))

    base_records: List[Dict[str, Any]] = []
    for row in rows:
        label = normalize_text(row.get("label"))
        if label not in {"A>B", "B>A", "Tie"}:
            continue
        sample = base_sample(
            sample_id="pending",
            dataset="judgebench",
            task_type="pairwise_bias",
            prompt=row.get("question", ""),
            answer_a=row.get("response_A", ""),
            answer_b=row.get("response_B", ""),
            human_label=label,
            human_score={
                "score_format": "pairwise_preference",
                "scoring_system": "pairwise_preference",
                "pairwise_preference": pairwise_label_score(label),
                "label": label,
            },
            metadata={
                "source": "ScalerLab/JudgeBench",
                "source_url": SOURCE_URLS["judgebench_claude"],
                "score_format": "pairwise_preference",
                "scoring_system": "pairwise_preference",
                "pair_id": normalize_text(row.get("pair_id")),
                "original_id": row.get("original_id"),
                "source_task": normalize_text(row.get("source")),
                "response_model": normalize_text(row.get("response_model")),
                "bias_type": "reasoning_difficulty",
                "perturbation_applied": "none",
                "actual_mapping": {"answer_a": "response_A", "answer_b": "response_B"},
            },
        )
        ok, _ = valid_sample(sample, require_pair=True)
        if ok:
            base_records.append(sample)

    # Keep all available JudgeBench pairs and construct multiple deterministic
    # perturbation variants from each item to support larger target sizes.
    selected_base = balanced_select(base_records, len(base_records))
    perturbed: List[Dict[str, Any]] = []
    bias_types = ("position", "length", "format", "rubric_sensitivity")

    def formatted(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        bullets = [f"- {s}" for s in sentences[:5] if s]
        if not bullets:
            return text
        return "Reformatted variant:\n" + "\n".join(bullets)

    for original in selected_base:
        for bias_type in bias_types:
            answer_a = original["answer_a"]
            answer_b = original["answer_b"]
            label = original["human_label"]
            prompt = original["prompt"]
            metadata = dict(original["metadata"])
            metadata.update(
                {
                    "source": "JudgeBench deterministic perturbation",
                    "parent_id": original["metadata"].get("pair_id"),
                    "bias_type": bias_type,
                    "perturbation_applied": bias_type,
                }
            )

            if bias_type == "position":
                answer_a, answer_b = answer_b, answer_a
                label = invert_label(label)
                metadata["actual_mapping"] = {"answer_a": "original_response_B", "answer_b": "original_response_A"}
            elif bias_type == "length":
                padding = (
                    "\n\nAdditional explanation: This answer presents the reasoning in several steps, "
                    "but the added framing is intentionally generic and should not change the "
                    "ground-truth preference."
                )
                if label == "A>B":
                    answer_b = answer_b + padding
                elif label == "B>A":
                    answer_a = answer_a + padding
                else:
                    answer_a = answer_a + padding
                metadata["actual_mapping"] = {"answer_a": "response_A", "answer_b": "response_B"}
            elif bias_type == "format":
                if label == "A>B":
                    answer_b = formatted(answer_b)
                elif label == "B>A":
                    answer_a = formatted(answer_a)
                else:
                    answer_b = formatted(answer_b)
                metadata["actual_mapping"] = {"answer_a": "response_A", "answer_b": "response_B"}
            else:
                prompt = (
                    f"{prompt}\n\nRubric (paraphrased): Prefer reliable reasoning and correct final "
                    "outcomes; do not reward verbosity or polished formatting by itself."
                )
                metadata["rubric_variant"] = (
                    "Prefer the answer with more reliable reasoning and correct final outcome; "
                    "do not reward verbosity or polished formatting alone."
                )
                metadata["actual_mapping"] = {"answer_a": "response_A", "answer_b": "response_B"}

            metadata["perturbation_variant_id"] = bias_type
            human_score = dict(original["human_score"])
            human_score.update(
                {
                    "score_format": "pairwise_preference",
                    "scoring_system": "pairwise_preference",
                    "pairwise_preference": pairwise_label_score(label),
                    "label": label,
                }
            )
            perturbed.append(
                base_sample(
                    sample_id="pending",
                    dataset="synthetic_perturbed",
                    task_type="pairwise_bias",
                    prompt=prompt,
                    answer_a=answer_a,
                    answer_b=answer_b,
                    human_label=label,
                    human_score=human_score,
                    metadata=metadata,
                )
            )

    perturb_target = min(len(perturbed), max(int(target * 2.0), 800))
    selected = selected_base + balanced_select(perturbed, perturb_target)
    RNG.shuffle(selected)
    for idx, sample in enumerate(selected, 1):
        sample["id"] = f"judge_bias_{idx:04d}"
    return selected


def split_sentences(text: str, limit: int = 4) -> List[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", normalize_text(text)) if len(s.strip()) > 12]
    return sentences[:limit]


def build_wikieval_samples(wikieval_path: Path) -> List[Dict[str, Any]]:
    rows = read_parquet(wikieval_path)
    samples: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        question = strip_prefix(normalize_text(row.get("question")), ["Question:"])
        answer = strip_prefix(normalize_text(row.get("answer")), ["Answer:"])
        context = normalize_text(row.get("context_v1"), max_chars=8000)
        source = normalize_text(row.get("source", ""))

        comparisons = [
            ("ungrounded_answer", "wikieval_grounded_vs_ungrounded", 1.0),
            ("poor_answer", "wikieval_grounded_vs_poor", 2.0),
        ]
        for answer_b_key, dataset_name, b_score in comparisons:
            answer_b = normalize_text(row.get(answer_b_key))
            sample = base_sample(
                sample_id="pending",
                dataset=dataset_name,
                task_type="factuality_rag",
                prompt=question,
                context=context,
                answer_a=answer,
                answer_b=answer_b,
                reference=answer,
                human_label="A>B",
                human_score={
                    "score_format": "pairwise_factuality_scores",
                    "scoring_system": "pairwise_factuality",
                    "pairwise_preference": pairwise_label_score("A>B"),
                    "label": "A>B",
                    "faithfulness_a": 5.0,
                    "faithfulness_b": b_score,
                    "answer_relevance_a": 5.0,
                    "answer_relevance_b": max(1.0, b_score),
                    "evidence_support_rate_a": 1.0,
                    "evidence_support_rate_b": 0.0 if answer_b_key == "ungrounded_answer" else 0.25,
                },
                metadata={
                    "source": "RAGAS WikiEval",
                    "source_url": SOURCE_URLS["wikieval"],
                    "score_format": "pairwise_factuality_scores",
                    "scoring_system": "pairwise_factuality",
                    "factuality_task_form": "pairwise",
                    "factuality_label": "supported",
                    "original_index": int(idx),
                    "wikieval_source": source,
                    "comparison_type": answer_b_key,
                    "answer_a_evidence_claims": [
                        {"claim": claim, "support": "Supported"} for claim in split_sentences(answer)
                    ],
                    "answer_b_evidence_claims": [
                        {"claim": claim, "support": "Refuted" if answer_b_key == "ungrounded_answer" else "NEI"}
                        for claim in split_sentences(answer_b)
                    ],
                },
            )
            ok, _ = valid_sample(sample, require_pair=True)
            if ok:
                samples.append(sample)
    return samples


def parse_binary_label(value: Any) -> Optional[int]:
    text = normalize_text(value).lower()
    if text in {"1", "1.0", "true", "yes", "supported", "relevant"}:
        return 1
    if text in {"0", "0.0", "false", "no", "unsupported", "irrelevant"}:
        return 0
    return None


def factuality_label_from_binary(
    context_rel: Optional[int],
    faithfulness: Optional[int],
    answer_rel: Optional[int],
) -> str:
    if faithfulness == 1 and answer_rel != 0 and context_rel != 0:
        return "supported"
    if faithfulness == 0 or answer_rel == 0 or context_rel == 0:
        return "unsupported"
    return "ambiguous"


def factuality_label_from_scores(*scores: Optional[float]) -> str:
    available = [score for score in scores if score is not None]
    if not available:
        return "ambiguous"
    avg = sum(available) / len(available)
    if avg >= 0.75:
        return "supported"
    if avg <= 0.40:
        return "unsupported"
    return "ambiguous"


def build_ares_samples(ares_path: Path, target: int = 300) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with ares_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query = normalize_text(row.get("Query") or row.get("input"))
            answer = normalize_text(row.get("Answer"))
            document = normalize_text(row.get("Document") or row.get("output"), max_chars=8000)
            if not query or not answer or not document:
                continue
            context_rel = parse_binary_label(row.get("Context_Relevance_Label"))
            faithfulness = parse_binary_label(row.get("Answer_Faithfulness_Label"))
            answer_rel = parse_binary_label(row.get("Answer_Relevance_Label"))
            available_labels = [
                label for label in (context_rel, faithfulness, answer_rel) if label is not None
            ]
            if not available_labels:
                continue

            avg = sum(available_labels) / len(available_labels)
            factuality_label = factuality_label_from_binary(context_rel, faithfulness, answer_rel)
            score_0_1 = round(avg, 4)
            score_1_5 = round(1.0 + avg * 4.0, 4)
            sample = base_sample(
                sample_id="pending",
                dataset="ares_nq",
                task_type="factuality_rag",
                prompt=query,
                context=document,
                answer_a=answer,
                answer_b="[SINGLE_ANSWER_FACTUALITY_TASK]",
                reference=normalize_text(row.get("output"), max_chars=4000),
                human_label=factuality_label,
                human_score={
                    "score_format": "single_answer_factuality_labels",
                    "scoring_system": "single_answer_factuality",
                    "factuality_label": factuality_label,
                    "factuality_label_score": factuality_label_score(factuality_label),
                    "context_relevance_label": context_rel,
                    "answer_faithfulness_label": faithfulness,
                    "answer_relevance_label": answer_rel,
                    "factuality_score_0_1": score_0_1,
                    "factuality_score_1_5": score_1_5,
                    "label": factuality_label,
                },
                metadata={
                    "source": "stanford-futuredata/ARES nq_labeled_output.tsv",
                    "source_url": SOURCE_URLS["ares_nq_labeled_api_raw"],
                    "score_format": "single_answer_factuality_labels",
                    "scoring_system": "single_answer_factuality",
                    "factuality_task_form": "single_answer",
                    "row_id": normalize_text(row.get("id")),
                    "wikipedia_id": normalize_text(row.get("wikipedia_id")),
                    "paragraph_number": normalize_text(row.get("paragraph_number")),
                    "factuality_label": factuality_label,
                    "context_relevance_label": context_rel,
                    "answer_relevance_label": answer_rel,
                    "atomic_facts": [
                        {
                            "fact": claim,
                            "support": "Supported"
                            if faithfulness == 1
                            else "Refuted/NEI"
                            if faithfulness == 0
                            else "Unknown",
                        }
                        for claim in split_sentences(answer, limit=3)
                    ],
                },
            )
            ok, _ = valid_sample(sample, require_pair=False)
            if ok:
                records.append(sample)

    selected = balanced_select(
        records,
        target,
        label_getter=lambda sample: sample["metadata"].get("factuality_label", "unknown"),
    )
    return selected


def build_factuality_rag(wikieval_path: Path, ares_path: Path, *, target: int) -> List[Dict[str, Any]]:
    wikieval_samples = build_wikieval_samples(wikieval_path)
    ares_samples = build_ares_samples(ares_path, target=max(int(target * 2.2), 900))
    selected = wikieval_samples + ares_samples
    RNG.shuffle(selected)
    for idx, sample in enumerate(selected, 1):
        sample["id"] = f"factuality_rag_{idx:04d}"
    return selected


def canonical_label(sample: Dict[str, Any]) -> str:
    return str(sample.get("human_label") or sample.get("metadata", {}).get("factuality_label") or "NA")


def split_group_key(sample: Dict[str, Any]) -> str:
    task_type = normalize_text(sample.get("task_type"))
    metadata = sample.get("metadata", {})
    prompt_anchor = normalize_text(sample.get("prompt"))

    if task_type == "pairwise_bias":
        anchor = normalize_text(metadata.get("parent_id") or metadata.get("pair_id"))
        if anchor:
            return f"{task_type}|{anchor}"

    if task_type == "factuality_rag":
        anchor = normalize_text(metadata.get("row_id") or metadata.get("original_index"))
        if anchor:
            return f"{task_type}|{anchor}"

    if prompt_anchor:
        return f"{task_type}|{prompt_anchor}"

    return f"{task_type}|{stable_hash(task_type, normalize_text(sample.get('id')))}"


def select_target_size(
    samples: List[Dict[str, Any]],
    target: int,
    *,
    label_getter,
) -> List[Dict[str, Any]]:
    if len(samples) < target:
        raise RuntimeError(f"not enough samples for target={target}; got {len(samples)}")
    return balanced_select(samples, target, label_getter=label_getter)


def assign_splits(samples: List[Dict[str, Any]]) -> None:
    # Group linked records (same prompt or perturbation parent) and assign each
    # group to a single split to avoid train/dev/test leakage.
    group_members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        group_members[split_group_key(sample)].append(sample)

    strata: Dict[Tuple[str, str, str], List[List[Dict[str, Any]]]] = defaultdict(list)
    for members in group_members.values():
        representative = members[0]
        dominant_label = Counter(canonical_label(sample) for sample in members).most_common(1)[0][0]
        strata[(representative["task_type"], representative["dataset"], dominant_label)].append(members)

    train_ratio, dev_ratio, _test_ratio = SPLIT_RATIOS
    for grouped_members in strata.values():
        RNG.shuffle(grouped_members)
        n_groups = len(grouped_members)
        train_cut = int(round(n_groups * train_ratio))
        dev_cut = train_cut + int(round(n_groups * dev_ratio))
        for idx, members in enumerate(grouped_members):
            split = "train" if idx < train_cut else "dev" if idx < dev_cut else "test"
            for sample in members:
                sample["split"] = split
                sample["metadata"]["split"] = split


def qc_and_finalize(
    samples: List[Dict[str, Any]],
    *,
    require_pair: bool,
) -> Tuple[List[Dict[str, Any]], Counter, List[Dict[str, Any]]]:
    reasons: Counter = Counter()
    valid: List[Dict[str, Any]] = []
    removed_entries: List[Dict[str, Any]] = []
    for sample in samples:
        ok, reason = valid_sample(sample, require_pair=require_pair)
        if ok:
            valid.append(sample)
        else:
            reasons[reason] += 1
            removed_entries.append(
                {
                    "reason": reason,
                    "task_type": sample.get("task_type"),
                    "dataset": sample.get("dataset"),
                    "id": sample.get("id"),
                }
            )

    deduped, removed, duplicate_entries = deduplicate(valid)
    if removed:
        reasons["duplicate"] += removed
    removed_entries.extend(duplicate_entries)
    return deduped, reasons, removed_entries


def dataset_wrapper(name: str, samples: List[Dict[str, Any]], sources: List[str]) -> Dict[str, Any]:
    return {
        "dataset_info": {
            "name": name,
            "schema": "BEA-Judge unified sample schema",
            "schema_version": "1.2",
            "missing_value_policy": "Required text fields must be non-empty; optional absent text fields use null, not empty strings.",
            "created_at": utc_now(),
            "seed": SEED,
            "sample_count": len(samples),
            "sources": sources,
            "fields": CORE_FIELDS + ["language", "split", "metadata"],
        },
        "samples": samples,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def apply_task_field_contracts(samples: List[Dict[str, Any]]) -> None:
    for sample in samples:
        task_type = normalize_text(sample.get("task_type"))
        contract = TASK_FIELD_CONTRACTS.get(
            task_type,
            {
                "context": "optional",
                "answer_b": "optional",
                "reference": "optional",
            },
        )
        metadata = sample.setdefault("metadata", {})
        missing_reason = dict(metadata.get("missing_reason") or {})

        for field in OPTIONAL_TEXT_FIELDS:
            value = sample.get(field)
            if isinstance(value, str) and not value.strip():
                sample[field] = None
                requirement = contract.get(field, "optional")
                if requirement == "required":
                    missing_reason[field] = "missing_required_field"
                else:
                    missing_reason[field] = "not_required_for_task"

        metadata["field_contract"] = dict(contract)
        metadata["null_normalization"] = "optional_empty_text_fields_use_null"
        if missing_reason:
            metadata["missing_reason"] = missing_reason
        elif "missing_reason" in metadata:
            metadata.pop("missing_reason")


def split_leakage_counts(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_group: Dict[str, set] = defaultdict(set)
    by_prompt: Dict[Tuple[str, str], set] = defaultdict(set)

    for sample in samples:
        split = sample.get("split", "")
        by_group[split_group_key(sample)].add(split)
        by_prompt[(sample.get("task_type", ""), normalize_text(sample.get("prompt")))].add(split)

    group_leaks = sum(1 for splits in by_group.values() if len(splits) > 1)
    prompt_leaks = sum(1 for splits in by_prompt.values() if len(splits) > 1)
    return {
        "group_leakage_count": group_leaks,
        "task_prompt_leakage_count": prompt_leaks,
        "group_count": len(by_group),
        "task_prompt_group_count": len(by_prompt),
    }


def write_split_index(samples: List[Dict[str, Any]]) -> None:
    payload = {
        "created_at": utc_now(),
        "seed": SEED,
        "sample_count": len(samples),
        "index": [
            {
                "id": sample.get("id"),
                "split": sample.get("split"),
                "task_type": sample.get("task_type"),
                "dataset": sample.get("dataset"),
                "group_key": split_group_key(sample),
            }
            for sample in samples
        ],
    }
    write_json(OUTPUT_DIR / "split_index.json", payload)


def write_core_subset(
    all_samples: List[Dict[str, Any]],
    subset_size: int,
    *,
    full_size: int,
) -> None:
    if len(all_samples) < subset_size:
        raise RuntimeError(f"Cannot write {subset_size} subset from {len(all_samples)} samples")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in all_samples:
        grouped[sample["task_type"]].append(sample)

    per_task = subset_size // len(grouped)
    selected: List[Dict[str, Any]] = []
    for task_type in sorted(grouped):
        task_samples = sorted(grouped[task_type], key=lambda sample: sample["id"])
        selected.extend(task_samples[:per_task])

    remainder = subset_size - len(selected)
    if remainder > 0:
        selected_ids = {sample["id"] for sample in selected}
        extras = [sample for sample in sorted(all_samples, key=lambda sample: sample["id"]) if sample["id"] not in selected_ids]
        selected.extend(extras[:remainder])

    wrapper = dataset_wrapper("BEA-Judge三类核心公开数据集（2400确定性子集）", selected, list(SOURCE_URLS.values()))
    wrapper["dataset_info"]["derived_from"] = f"bea_judge_core_{full_size}.json"
    wrapper["dataset_info"]["subset_rule"] = "deterministic first-N per task_type after seeded build"
    write_json(PROCESSED_DIR / f"bea_judge_core_{subset_size}.json", wrapper)


def stats_for(samples: List[Dict[str, Any]], qc_removed: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    def avg(values: List[int]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    def text_len(value: Any) -> int:
        return len(normalize_text(value))

    human_labels = [s.get("human_label") for s in samples if s.get("human_label")]
    tie_count = sum(1 for label in human_labels if label == "Tie")
    tie_ratio = round(tie_count / len(human_labels), 4) if human_labels else 0.0

    return {
        "created_at": utc_now(),
        "seed": SEED,
        "total_samples": len(samples),
        "by_task_type": dict(Counter(s["task_type"] for s in samples)),
        "by_dataset": dict(Counter(s["dataset"] for s in samples)),
        "by_split": dict(Counter(s.get("split", "NA") for s in samples)),
        "by_language": dict(Counter(s.get("language", "unknown") for s in samples)),
        "by_score_format": dict(Counter(s.get("metadata", {}).get("score_format", "unknown") for s in samples)),
        "by_scoring_system": dict(Counter(s.get("metadata", {}).get("scoring_system", "unknown") for s in samples)),
        "by_factuality_task_form": dict(
            Counter(
                s.get("metadata", {}).get("factuality_task_form", "NA")
                for s in samples
                if s.get("task_type") == "factuality_rag"
            )
        ),
        "human_label_distribution": dict(Counter(str(s.get("human_label")) for s in samples if s.get("human_label"))),
        "tie_ratio_among_pairwise_labels": tie_ratio,
        "factuality_label_distribution": dict(
            Counter(
                s.get("metadata", {}).get("factuality_label")
                for s in samples
                if s.get("metadata", {}).get("factuality_label")
            )
        ),
        "avg_chars": {
            "prompt": avg([text_len(s.get("prompt")) for s in samples]),
            "context": avg([text_len(s.get("context")) for s in samples]),
            "answer_a": avg([text_len(s.get("answer_a")) for s in samples]),
            "answer_b": avg([text_len(s.get("answer_b")) for s in samples if s.get("answer_b")]),
        },
        "qc_removed": qc_removed,
        "split_leakage": split_leakage_counts(samples),
        "required_fields": CORE_FIELDS,
    }


def write_splits(all_samples: List[Dict[str, Any]]) -> None:
    for split in ("train", "dev", "test"):
        split_samples = [sample for sample in all_samples if sample.get("split") == split]
        write_json(
            SPLIT_DIR / f"{split}.json",
            dataset_wrapper(f"BEA-Judge {split} split", split_samples, list(SOURCE_URLS.values())),
        )


def acquire_sources() -> Dict[str, Path]:
    print("Acquiring public source files...")
    paths = {
        "mt_bench_human": RAW_DIR / "mt_bench_human.parquet",
        "pandalm_test": RAW_DIR / "pandalm_testset_v1.json",
        "judgebench_claude": RAW_DIR / "judgebench_claude.jsonl",
        "wikieval": RAW_DIR / "wikieval.parquet",
        "ares_nq_labeled": RAW_DIR / "ares_nq_labeled_output.tsv",
    }

    download_file(SOURCE_URLS["mt_bench_human"], paths["mt_bench_human"])
    download_github_api_base64(SOURCE_URLS["pandalm_test_api"], paths["pandalm_test"])
    download_file(SOURCE_URLS["judgebench_claude"], paths["judgebench_claude"])
    download_file(SOURCE_URLS["wikieval"], paths["wikieval"])
    download_file(
        SOURCE_URLS["ares_nq_labeled_api_raw"],
        paths["ares_nq_labeled"],
        headers={"Accept": "application/vnd.github.raw"},
        timeout=240,
    )

    manifest = {
        "created_at": utc_now(),
        "sources": {
            key: {
                "url": SOURCE_URLS.get(key) or SOURCE_URLS.get(f"{key}_api") or SOURCE_URLS.get(f"{key}_api_raw"),
                "path": str(path.relative_to(PROJECT_ROOT)),
                "absolute_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in paths.items()
        },
    }
    write_json(OUTPUT_DIR / "data_manifest.json", manifest)
    return paths


def assert_count(name: str, samples: List[Dict[str, Any]], expected: int) -> None:
    if len(samples) != expected:
        raise RuntimeError(f"{name} expected {expected} samples, got {len(samples)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BEA-Judge unified datasets.")
    parser.add_argument(
        "--target-per-task",
        type=int,
        default=DEFAULT_TARGET_PER_TASK,
        help=f"Target sample count per task type (default: {DEFAULT_TARGET_PER_TASK})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_per_task = int(args.target_per_task)
    if target_per_task <= 0:
        raise RuntimeError("--target-per-task must be a positive integer")

    targets = {
        "open_qa": target_per_task,
        "judge_bias": target_per_task,
        "factuality_rag": target_per_task,
    }

    ensure_dirs()
    paths = acquire_sources()

    print("Building open-ended answer quality set...")
    open_qa_pool, open_qc, open_removed = qc_and_finalize(
        build_open_qa(paths["mt_bench_human"], paths["pandalm_test"]),
        require_pair=True,
    )
    open_qa = select_target_size(open_qa_pool, targets["open_qa"], label_getter=lambda sample: canonical_label(sample))
    assert_count("open_qa", open_qa, targets["open_qa"])

    print("Building judge-bias and difficult-pair set...")
    judge_pool, judge_qc, judge_removed = qc_and_finalize(
        build_judge_bias(paths["judgebench_claude"], target=targets["judge_bias"]),
        require_pair=True,
    )
    judge_bias = select_target_size(judge_pool, targets["judge_bias"], label_getter=lambda sample: canonical_label(sample))
    assert_count("judge_bias", judge_bias, targets["judge_bias"])

    print("Building factuality and RAG set...")
    factuality_pool, factuality_qc, factuality_removed = qc_and_finalize(
        build_factuality_rag(paths["wikieval"], paths["ares_nq_labeled"], target=targets["factuality_rag"]),
        require_pair=False,
    )
    factuality_rag = select_target_size(
        factuality_pool,
        targets["factuality_rag"],
        label_getter=lambda sample: sample.get("metadata", {}).get("factuality_label")
        or sample.get("dataset")
        or canonical_label(sample),
    )
    assert_count("factuality_rag", factuality_rag, targets["factuality_rag"])

    all_samples = open_qa + judge_bias + factuality_rag
    assign_splits(all_samples)

    write_json(
        PROCESSED_DIR / "open_qa_dataset.json",
        dataset_wrapper(
            "开放式回答质量数据集",
            open_qa,
            [SOURCE_URLS["mt_bench_human"], SOURCE_URLS["pandalm_test_api"]],
        ),
    )
    write_json(
        PROCESSED_DIR / "judge_bias_dataset.json",
        dataset_wrapper(
            "Judge偏差与困难样本数据集",
            judge_bias,
            [SOURCE_URLS["judgebench_claude"], "constructed perturbations from JudgeBench"],
        ),
    )
    write_json(
        PROCESSED_DIR / "factuality_rag_dataset.json",
        dataset_wrapper(
            "事实性与RAG数据集",
            factuality_rag,
            [SOURCE_URLS["wikieval"], SOURCE_URLS["ares_nq_labeled_api_raw"]],
        ),
    )
    write_json(
        PROCESSED_DIR / f"bea_judge_core_{len(all_samples)}.json",
        dataset_wrapper("BEA-Judge三类核心公开数据集", all_samples, list(SOURCE_URLS.values())),
    )
    write_json(
        PROCESSED_DIR / "bea_judge_core_latest.json",
        dataset_wrapper("BEA-Judge三类核心公开数据集", all_samples, list(SOURCE_URLS.values())),
    )
    if len(all_samples) >= 1200:
        write_core_subset(all_samples, 1200, full_size=len(all_samples))
    write_splits(all_samples)
    write_split_index(all_samples)

    qc_removed = {
        "open_qa": dict(open_qc),
        "judge_bias": dict(judge_qc),
        "factuality_rag": dict(factuality_qc),
    }
    write_json(
        OUTPUT_DIR / "qc_removed_log.json",
        {
            "created_at": utc_now(),
            "entries": {
                "open_qa": open_removed,
                "judge_bias": judge_removed,
                "factuality_rag": factuality_removed,
            },
        },
    )
    statistics = stats_for(all_samples, qc_removed)
    write_json(OUTPUT_DIR / "dataset_statistics.json", statistics)

    print("Done.")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
