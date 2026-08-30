"""Run external lightweight baselines for BEA-Judge pairwise comparison."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_adapter import samples_from_payload  # noqa: E402


PAIRWISE_LABELS = ("A>B", "B>A", "Tie")
LABEL_TO_RESULT = {"A>B": "[RESULT] A", "B>A": "[RESULT] B", "Tie": "[RESULT] Tie"}
PROMETHEUS2_LABEL_COMPLETIONS = {"A>B": "A", "B>A": "B"}
GLIDER_SCORE_COMPLETIONS = {"A>B": "1", "Tie": "2", "B>A": "3"}
RESULT_TO_LABEL = {value: key for key, value in LABEL_TO_RESULT.items()}


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_pairwise_samples(dataset: Path, split: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    samples = samples_from_payload(read_json(dataset))
    rows = [
        sample
        for sample in samples
        if sample.get("split") == split
        and sample.get("human_label") in PAIRWISE_LABELS
        and normalize_text(sample.get("answer_a"))
        and normalize_text(sample.get("answer_b"))
    ]
    return rows[:limit] if limit is not None else rows


def stratified_smoke_samples(dataset: Path, split: str, per_label: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts = Counter()
    for sample in load_pairwise_samples(dataset, split):
        label = str(sample.get("human_label"))
        if label in PAIRWISE_LABELS and counts[label] < per_label:
            selected.append(sample)
            counts[label] += 1
        if all(counts[label] >= per_label for label in PAIRWISE_LABELS):
            break
    return selected


def softmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    peak = max(scores)
    values = [math.exp(score - peak) for score in scores]
    total = sum(values)
    return [value / total for value in values]


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    scores: List[float] = []
    for label in PAIRWISE_LABELS:
        tp = sum(actual == label and pred == label for actual, pred in zip(y_true, y_pred))
        fp = sum(actual != label and pred == label for actual, pred in zip(y_true, y_pred))
        fn = sum(actual == label and pred != label for actual, pred in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores))


def ece_score(y_true: Sequence[str], y_pred: Sequence[str], confidences: Sequence[float], bins: int = 10) -> float:
    if not y_true:
        return 0.0
    total = len(y_true)
    ece = 0.0
    for index in range(bins):
        lo = index / bins
        hi = (index + 1) / bins
        selected = [
            i
            for i, confidence in enumerate(confidences)
            if confidence >= lo and (confidence < hi if index < bins - 1 else confidence <= hi)
        ]
        if not selected:
            continue
        accuracy = sum(y_true[i] == y_pred[i] for i in selected) / len(selected)
        avg_confidence = sum(confidences[i] for i in selected) / len(selected)
        ece += len(selected) / total * abs(accuracy - avg_confidence)
    return float(ece)


def metrics_for_predictions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("pred_label") in PAIRWISE_LABELS]
    y_true = [str(row.get("gold_label")) for row in valid]
    y_pred = [str(row.get("pred_label")) for row in valid]
    confidences = [float(row.get("confidence", 0.0) or 0.0) for row in valid]
    tie_indices = [i for i, label in enumerate(y_true) if label == "Tie"]
    return {
        "n": len(rows),
        "valid_n": len(valid),
        "accuracy": round(sum(actual == pred for actual, pred in zip(y_true, y_pred)) / len(y_true), 6)
        if y_true
        else None,
        "macro_f1": round(macro_f1(y_true, y_pred), 6) if y_true else None,
        "ece": round(ece_score(y_true, y_pred, confidences), 6) if y_true else None,
        "tie_recall": round(sum(y_pred[i] == "Tie" for i in tie_indices) / len(tie_indices), 6)
        if tie_indices
        else None,
        "parse_failure_rate": round(1.0 - len(valid) / len(rows), 6) if rows else 1.0,
        "pred_label_distribution": dict(Counter(row.get("pred_label") for row in valid)),
        "gold_label_distribution": dict(Counter(y_true)),
        "total_inference_seconds": round(sum(float(row.get("inference_seconds", 0.0) or 0.0) for row in rows), 3),
    }


def grm_probabilities(score_a: float, score_b: float, margin: float) -> Dict[str, float]:
    diff = score_a - score_b
    win_probs = softmax([diff, -diff])
    tie_strength = max(0.0, 1.0 - abs(diff) / max(margin, 1e-6)) if margin > 0 else 0.0
    raw = {
        "A>B": win_probs[0] * (1.0 - 0.5 * tie_strength),
        "B>A": win_probs[1] * (1.0 - 0.5 * tie_strength),
        "Tie": tie_strength,
    }
    total = sum(raw.values())
    return {label: float(value / total) for label, value in raw.items()}


def grm_label_from_scores(score_a: float, score_b: float, margin: float) -> Tuple[str, Dict[str, float]]:
    diff = score_a - score_b
    if diff > margin:
        label = "A>B"
    elif -diff > margin:
        label = "B>A"
    else:
        label = "Tie"
    return label, grm_probabilities(score_a, score_b, margin)


def select_grm_margin(scored_rows: Sequence[Mapping[str, Any]]) -> Tuple[float, List[Dict[str, Any]]]:
    diffs = sorted(abs(float(row["score_a"]) - float(row["score_b"])) for row in scored_rows)
    if not diffs:
        return 0.0, []
    max_index = max(0, int((len(diffs) - 1) * 0.5))
    quantile_indices = sorted(set(round(max_index * step / 20) for step in range(21)))
    margins = [diffs[index] for index in quantile_indices]
    candidates: List[Dict[str, Any]] = []
    for margin in margins:
        pred_rows = []
        for row in scored_rows:
            pred, probs = grm_label_from_scores(float(row["score_a"]), float(row["score_b"]), float(margin))
            pred_rows.append(
                {
                    "gold_label": row["gold_label"],
                    "pred_label": pred,
                    "confidence": probs[pred],
                }
            )
        metrics = metrics_for_predictions(pred_rows)
        candidates.append({"margin": float(margin), "metrics": metrics})
    best = max(
        candidates,
        key=lambda row: (
            float(row["metrics"].get("macro_f1") or 0.0),
            float(row["metrics"].get("accuracy") or 0.0),
            float(row["metrics"].get("tie_recall") or 0.0),
            -float(row["metrics"].get("ece") or 1.0),
        ),
    )
    return float(best["margin"]), candidates


def build_pairwise_prompt(sample: Mapping[str, Any]) -> str:
    instruction = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference")) or "No reference answer is provided."
    if context:
        instruction = f"{instruction}\n\nContext:\n{context}"
    return "\n".join(
        [
            "You are an impartial evaluator. Choose the better response for the user instruction.",
            "Return exactly one of [RESULT] A, [RESULT] B, or [RESULT] Tie.",
            "",
            "Instruction:",
            instruction,
            "",
            "Response A:",
            normalize_text(sample.get("answer_a")),
            "",
            "Response B:",
            normalize_text(sample.get("answer_b")),
            "",
            "Reference:",
            reference,
            "",
            "Final answer:",
        ]
    )


def reward_messages(sample: Mapping[str, Any], answer_key: str) -> List[Dict[str, str]]:
    instruction = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference"))
    answer = normalize_text(sample.get(answer_key))
    pieces = ["Instruction:", instruction]
    if context:
        pieces.extend(["Context:", context])
    if reference:
        pieces.extend(["Reference:", reference])
    return [
        {"role": "user", "content": "\n".join(pieces)},
        {"role": "assistant", "content": answer},
    ]


def build_reward_text(sample: Mapping[str, Any], answer_key: str, tokenizer: Optional[Any] = None) -> str:
    messages = reward_messages(sample, answer_key)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return "\n\n".join(f"{message['role'].title()}: {message['content']}" for message in messages)


def build_qwen_prompt(sample: Mapping[str, Any], tokenizer: Optional[Any] = None) -> str:
    prompt = build_pairwise_prompt(sample)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def build_prometheus2_prompt(sample: Mapping[str, Any], tokenizer: Optional[Any] = None) -> str:
    instruction = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference")) or "No reference answer is provided."
    if context:
        instruction = f"{instruction}\n\nContext:\n{context}"
    prompt = "\n".join(
        [
            "You are Prometheus 2, a fair evaluator for pairwise response ranking.",
            "Compare Response A and Response B for the user instruction and reference answer.",
            "Return exactly one letter: A if Response A is better, or B if Response B is better.",
            "",
            "Instruction:",
            instruction,
            "",
            "Reference answer:",
            reference,
            "",
            "Response A:",
            normalize_text(sample.get("answer_a")),
            "",
            "Response B:",
            normalize_text(sample.get("answer_b")),
            "",
            "Final verdict:",
        ]
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def build_glider_prompt(sample: Mapping[str, Any], tokenizer: Optional[Any] = None) -> str:
    instruction = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference")) or "No reference answer is provided."
    if context:
        instruction = f"{instruction}\n\nContext:\n{context}"
    prompt = "\n".join(
        [
            "You are GLIDER, an evaluation model. Evaluate which response better satisfies the pass criteria.",
            "",
            "Data:",
            f"Instruction: {instruction}",
            f"Reference answer: {reference}",
            f"Response A: {normalize_text(sample.get('answer_a'))}",
            f"Response B: {normalize_text(sample.get('answer_b'))}",
            "",
            "Pass criteria:",
            "The better response should be more helpful, factually grounded in the instruction and reference, "
            "and avoid unsupported or misleading content.",
            "",
            "Rubric:",
            "Score 1: Response A is better than Response B.",
            "Score 2: Response A and Response B are approximately tied.",
            "Score 3: Response B is better than Response A.",
            "",
            "Return only the score number.",
            "Score:",
        ]
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def prediction_row(
    *,
    sample: Mapping[str, Any],
    pred_label: Optional[str],
    probabilities: Mapping[str, float],
    model_name: str,
    model_kind: str,
    inference_seconds: float,
    parse_status: str = "ok",
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    confidence = float(probabilities.get(str(pred_label), 0.0)) if pred_label in PAIRWISE_LABELS else 0.0
    return {
        "id": sample.get("id"),
        "dataset": sample.get("dataset"),
        "split": sample.get("split"),
        "gold_label": sample.get("human_label"),
        "pred_label": pred_label,
        "label_probabilities": {label: round(float(probabilities.get(label, 0.0)), 6) for label in PAIRWISE_LABELS},
        "confidence": round(confidence, 6),
        "parse_status": parse_status,
        "model_name": model_name,
        "model_kind": model_kind,
        "inference_seconds": round(float(inference_seconds), 6),
        "extra": dict(extra or {}),
    }


def load_transformers_model(model_path: Path, model_kind: str, load_in_4bit: bool, device_map: str) -> Tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing transformers/torch dependencies in the active environment.") from exc

    if not model_path.exists():
        raise FileNotFoundError(
            f"model path does not exist: {model_path}. Download or place the model there before inference."
        )
    trust_remote_code = model_kind == "glider_evaluator"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        if model_kind != "glider_evaluator":
            raise
        trust_remote_code = False
        print(
            "Falling back to built-in Phi3 tokenizer/config for GLIDER because local custom code is unavailable.",
            file=sys.stderr,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            use_fast=True,
            local_files_only=True,
            trust_remote_code=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "local_files_only": True,
    }
    if load_in_4bit and model_kind in {"qwen_instruct", "prometheus2_pairwise", "glider_evaluator"}:
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except Exception:
            pass
    if model_kind == "grm_reward":
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path), **kwargs)
    elif model_kind in {"qwen_instruct", "prometheus2_pairwise", "glider_evaluator"}:
        kwargs["trust_remote_code"] = trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **kwargs)
    else:
        raise ValueError(f"unsupported model kind: {model_kind}")
    model.eval()
    return tokenizer, model


def scalar_reward(model: Any, tokenizer: Any, text: str) -> float:
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    return float(logits.reshape(-1)[-1].detach().float().cpu().item())


def run_grm_scoring(samples: Sequence[Mapping[str, Any]], model_path: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    tokenizer, model = load_transformers_model(model_path, "grm_reward", args.load_in_4bit, args.device_map)
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        start = time.time()
        score_a = scalar_reward(model, tokenizer, build_reward_text(sample, "answer_a", tokenizer))
        score_b = scalar_reward(model, tokenizer, build_reward_text(sample, "answer_b", tokenizer))
        rows.append(
            {
                "id": sample.get("id"),
                "dataset": sample.get("dataset"),
                "split": sample.get("split"),
                "gold_label": sample.get("human_label"),
                "score_a": score_a,
                "score_b": score_b,
                "inference_seconds": time.time() - start,
            }
        )
    return rows


def apply_grm_margin(scored_rows: Sequence[Mapping[str, Any]], margin: float, model_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in scored_rows:
        pred_label, probs = grm_label_from_scores(float(row["score_a"]), float(row["score_b"]), margin)
        sample = {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "human_label": row.get("gold_label"),
        }
        rows.append(
            prediction_row(
                sample=sample,
                pred_label=pred_label,
                probabilities=probs,
                model_name=model_name,
                model_kind="grm_reward",
                inference_seconds=float(row.get("inference_seconds", 0.0) or 0.0),
                extra={"score_a": row["score_a"], "score_b": row["score_b"], "tie_margin": margin},
            )
        )
    return rows


def apply_pairwise_margin(
    scored_rows: Sequence[Mapping[str, Any]],
    margin: float,
    model_name: str,
    model_kind: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in scored_rows:
        pred_label, probs = grm_label_from_scores(float(row["score_a"]), float(row["score_b"]), margin)
        sample = {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "human_label": row.get("gold_label"),
        }
        rows.append(
            prediction_row(
                sample=sample,
                pred_label=pred_label,
                probabilities=probs,
                model_name=model_name,
                model_kind=model_kind,
                inference_seconds=float(row.get("inference_seconds", 0.0) or 0.0),
                extra={
                    "score_a": row["score_a"],
                    "score_b": row["score_b"],
                    "tie_margin": margin,
                    "tie_margin_source": "dev_only",
                },
            )
        )
    return rows


def label_log_likelihood(model: Any, tokenizer: Any, prompt: str, completion: str) -> float:
    import torch

    full_text = prompt + completion
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    full = tokenizer(full_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048).to(
        model.device
    )
    prompt_len = int(prompt_ids.shape[1])
    input_ids = full.input_ids
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    with torch.no_grad():
        outputs = model(**full)
        logits = outputs.logits[:, :-1, :]
        target = labels[:, 1:]
        mask = target != -100
        log_probs = torch.log_softmax(logits, dim=-1)
        token_scores = log_probs.gather(-1, target.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return float(token_scores[mask].sum().detach().float().cpu().item())


def run_qwen_predictions(samples: Sequence[Mapping[str, Any]], model_path: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    tokenizer, model = load_transformers_model(model_path, "qwen_instruct", args.load_in_4bit, args.device_map)
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        start = time.time()
        prompt = build_qwen_prompt(sample, tokenizer)
        scores = [label_log_likelihood(model, tokenizer, prompt, LABEL_TO_RESULT[label]) for label in PAIRWISE_LABELS]
        probs_list = softmax(scores)
        probs = {label: probs_list[index] for index, label in enumerate(PAIRWISE_LABELS)}
        pred_label = max(probs, key=probs.get)
        rows.append(
            prediction_row(
                sample=sample,
                pred_label=pred_label,
                probabilities=probs,
                model_name=args.model_name,
                model_kind="qwen_instruct",
                inference_seconds=time.time() - start,
                extra={"prompt_template": "pairwise_result_likelihood_v1"},
            )
        )
    return rows


def run_prometheus2_scoring(
    samples: Sequence[Mapping[str, Any]],
    model_path: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    tokenizer, model = load_transformers_model(model_path, "prometheus2_pairwise", args.load_in_4bit, args.device_map)
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        start = time.time()
        prompt = build_prometheus2_prompt(sample, tokenizer)
        score_a = label_log_likelihood(model, tokenizer, prompt, PROMETHEUS2_LABEL_COMPLETIONS["A>B"])
        score_b = label_log_likelihood(model, tokenizer, prompt, PROMETHEUS2_LABEL_COMPLETIONS["B>A"])
        rows.append(
            {
                "id": sample.get("id"),
                "dataset": sample.get("dataset"),
                "split": sample.get("split"),
                "gold_label": sample.get("human_label"),
                "score_a": score_a,
                "score_b": score_b,
                "inference_seconds": time.time() - start,
            }
        )
    return rows


def run_glider_predictions(samples: Sequence[Mapping[str, Any]], model_path: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
    tokenizer, model = load_transformers_model(model_path, "glider_evaluator", args.load_in_4bit, args.device_map)
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        start = time.time()
        prompt = build_glider_prompt(sample, tokenizer)
        scores = [
            label_log_likelihood(model, tokenizer, prompt, GLIDER_SCORE_COMPLETIONS[label])
            for label in PAIRWISE_LABELS
        ]
        probs_list = softmax(scores)
        probs = {label: probs_list[index] for index, label in enumerate(PAIRWISE_LABELS)}
        pred_label = max(probs, key=probs.get)
        rows.append(
            prediction_row(
                sample=sample,
                pred_label=pred_label,
                probabilities=probs,
                model_name=args.model_name,
                model_kind="glider_evaluator",
                inference_seconds=time.time() - start,
                extra={"prompt_template": "glider_pairwise_rubric_likelihood_v1"},
            )
        )
    return rows


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def comparison_rows(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for name, payload in report.get("baselines", {}).items():
        metrics = payload.get("test_metrics", {})
        rows.append(
            {
                "system": name,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "ece": metrics.get("ece"),
                "tie_recall": metrics.get("tie_recall"),
                "parse_failure_rate": metrics.get("parse_failure_rate"),
                "n": metrics.get("n"),
            }
        )
    return rows


def run_grm(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    dev_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "dev", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "dev", args.limit)
    test_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "test", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "test", args.limit)
    model_path = resolve_root_path(args.model_path)
    dev_scored = run_grm_scoring(dev_samples, model_path, args)
    margin, candidates = select_grm_margin(dev_scored)
    dev_rows = apply_grm_margin(dev_scored, margin, args.model_name)
    test_scored = run_grm_scoring(test_samples, model_path, args)
    test_rows = apply_grm_margin(test_scored, margin, args.model_name)
    write_jsonl(output_dir / "grm_dev_predictions.jsonl", dev_rows)
    write_jsonl(output_dir / "grm_test_predictions.jsonl", test_rows)
    return {
        "model_name": args.model_name,
        "model_kind": "grm_reward",
        "model_path": str(model_path),
        "selected_margin": margin,
        "margin_candidates": candidates,
        "dev_metrics": metrics_for_predictions(dev_rows),
        "test_metrics": metrics_for_predictions(test_rows),
    }


def run_qwen(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    dev_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "dev", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "dev", args.limit)
    test_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "test", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "test", args.limit)
    model_path = resolve_root_path(args.model_path)
    dev_rows = run_qwen_predictions(dev_samples, model_path, args)
    test_rows = run_qwen_predictions(test_samples, model_path, args)
    write_jsonl(output_dir / "qwen_dev_predictions.jsonl", dev_rows)
    write_jsonl(output_dir / "qwen_test_predictions.jsonl", test_rows)
    return {
        "model_name": args.model_name,
        "model_kind": "qwen_instruct",
        "model_path": str(model_path),
        "dev_metrics": metrics_for_predictions(dev_rows),
        "test_metrics": metrics_for_predictions(test_rows),
    }


def run_prometheus2(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    dev_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "dev", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "dev", args.limit)
    test_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "test", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "test", args.limit)
    model_path = resolve_root_path(args.model_path)
    dev_scored = run_prometheus2_scoring(dev_samples, model_path, args)
    margin, candidates = select_grm_margin(dev_scored)
    dev_rows = apply_pairwise_margin(dev_scored, margin, args.model_name, "prometheus2_pairwise")
    test_scored = run_prometheus2_scoring(test_samples, model_path, args)
    test_rows = apply_pairwise_margin(test_scored, margin, args.model_name, "prometheus2_pairwise")
    write_jsonl(output_dir / "prometheus2_dev_predictions.jsonl", dev_rows)
    write_jsonl(output_dir / "prometheus2_test_predictions.jsonl", test_rows)
    return {
        "model_name": args.model_name,
        "model_kind": "prometheus2_pairwise",
        "model_path": str(model_path),
        "selected_margin": margin,
        "margin_candidates": candidates,
        "dev_metrics": metrics_for_predictions(dev_rows),
        "test_metrics": metrics_for_predictions(test_rows),
    }


def run_glider(args: argparse.Namespace, output_dir: Path) -> Dict[str, Any]:
    dev_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "dev", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "dev", args.limit)
    test_samples = stratified_smoke_samples(resolve_root_path(args.dataset), "test", args.smoke_per_label) if args.smoke else load_pairwise_samples(resolve_root_path(args.dataset), "test", args.limit)
    model_path = resolve_root_path(args.model_path)
    dev_rows = run_glider_predictions(dev_samples, model_path, args)
    test_rows = run_glider_predictions(test_samples, model_path, args)
    write_jsonl(output_dir / "glider_dev_predictions.jsonl", dev_rows)
    write_jsonl(output_dir / "glider_test_predictions.jsonl", test_rows)
    return {
        "model_name": args.model_name,
        "model_kind": "glider_evaluator",
        "model_path": str(model_path),
        "dev_metrics": metrics_for_predictions(dev_rows),
        "test_metrics": metrics_for_predictions(test_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external lightweight baselines.")
    parser.add_argument("--dataset", default="datasets/processed/bea_judge_cleaned_10000.json")
    parser.add_argument(
        "--model-kind",
        choices=["grm_reward", "qwen_instruct", "prometheus2_pairwise", "glider_evaluator"],
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="datasets/model_outputs/external_3b_baseline_comparison")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-label", type=int, default=10)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_root_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if args.model_kind == "grm_reward":
        baseline = run_grm(args, output_dir)
    elif args.model_kind == "qwen_instruct":
        baseline = run_qwen(args, output_dir)
    elif args.model_kind == "prometheus2_pairwise":
        baseline = run_prometheus2(args, output_dir)
    else:
        baseline = run_glider(args, output_dir)

    report_path = output_dir / "external_3b_baseline_comparison_report.json"
    existing = read_json(report_path) if report_path.exists() else {"baselines": {}}
    existing.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    existing.setdefault("baselines", {})
    existing["baselines"][args.model_name] = baseline
    existing["updated_at"] = datetime.now(timezone.utc).isoformat()
    existing["runtime_seconds_last_run"] = round(time.time() - started, 3)
    report_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    table_rows = comparison_rows(existing)
    (output_dir / "external_3b_baseline_comparison_table.md").write_text(
        markdown_table(table_rows),
        encoding="utf-8",
    )
    print(json.dumps(baseline["test_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
