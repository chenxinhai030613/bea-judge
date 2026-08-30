from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from dataset_adapter import adapt_dataset_payload


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
PROCESSED = DATASETS / "processed"
EVAL_OUT = DATASETS / "judge_outputs"
DEFAULT_JUDGE_CONFIG_PATH = ROOT / "configs" / "judge.json"
DEFAULT_M_PROMETHEUS_MODEL_PATH = "Unbabel/M-Prometheus-3B"
DEFAULT_PROMETHEUS2_MODEL_PATH = "prometheus-eval/prometheus-8x7b-v2.0"


PAIRWISE_LABELS = ("A>B", "B>A", "Tie")
DIRECT_DIMENSIONS = (
    "relevance",
    "completeness",
    "factuality",
    "instruction_following",
    "clarity",
    "safety",
)


@dataclass(frozen=True)
class JudgeConfig:
    name: str = "m_prometheus_3b_base"
    version: str = DEFAULT_M_PROMETHEUS_MODEL_PATH
    temperature: float = 0.0
    top_p: float = 1.0
    mode: str = "pairwise"
    prompt_template: str = "m_prometheus_pairwise_v1"
    backend: str = "m_prometheus"
    model_path: str = DEFAULT_M_PROMETHEUS_MODEL_PATH
    max_new_tokens: int = 256
    device: str = "auto"
    allow_fallback: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dirs() -> None:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)


def read_dataset(path: Path) -> Dict[str, Any]:
    return adapt_dataset_payload(json.loads(path.read_text(encoding="utf-8")))


def load_judge_config(path: Path = DEFAULT_JUDGE_CONFIG_PATH) -> JudgeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return JudgeConfig(
        name=str(payload.get("name", JudgeConfig.name)),
        version=str(payload.get("version", JudgeConfig.version)),
        temperature=float(payload.get("temperature", JudgeConfig.temperature)),
        top_p=float(payload.get("top_p", JudgeConfig.top_p)),
        mode=str(payload.get("mode", JudgeConfig.mode)),
        prompt_template=str(payload.get("prompt_template", JudgeConfig.prompt_template)),
        backend=str(payload.get("backend", JudgeConfig.backend)),
        model_path=str(payload.get("model_path", JudgeConfig.model_path)),
        max_new_tokens=int(payload.get("max_new_tokens", JudgeConfig.max_new_tokens)),
        device=str(payload.get("device", JudgeConfig.device)),
        allow_fallback=bool(payload.get("allow_fallback", JudgeConfig.allow_fallback)),
    )


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", str(text)).strip()


def score_from_token(token: str) -> Optional[float]:
    token = token.upper().strip()
    mapping = {"A>B": 1.0, "B>A": -1.0, "TIE": 0.0}
    return mapping.get(token)


def extract_label(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    raw = normalize_text(text)
    label_match = re.search(r"\b(A>B|B>A|Tie)\b", raw, flags=re.IGNORECASE)
    score_match = re.search(r"(-?\d+(?:\.\d+)?)", raw)
    parsed: Dict[str, Any] = {"raw": raw}
    if label_match:
        label = label_match.group(1)
        parsed["label"] = label if label != "tie" else "Tie"
        return parsed["label"], parsed
    if score_match:
        value = float(score_match.group(1))
        if value > 0.25:
            parsed["label"] = "A>B"
            return "A>B", parsed
        if value < -0.25:
            parsed["label"] = "B>A"
            return "B>A", parsed
        parsed["label"] = "Tie"
        return "Tie", parsed
    return None, parsed


PROMETHEUS_PAIRWISE_RUBRIC = (
    "Evaluate which response better satisfies the user instruction. Consider relevance, "
    "completeness, factuality, instruction following, clarity, and safety. Use the "
    "reference answer and context when available. Return exactly one final result marker: "
    "[RESULT] A, [RESULT] B, or [RESULT] Tie."
)
PROMETHEUS_DIRECT_RUBRIC = (
    "Score the response on six dimensions: relevance, completeness, factuality, "
    "instruction_following, clarity, and safety. Each dimension must be an integer "
    "from 1 to 5, where 1 is poor and 5 is excellent. Also provide overall_score "
    "from 1 to 5."
)
DEFAULT_PROMETHEUS_JUDGE_IDENTITY = "M-Prometheus-3B"


def build_prometheus_pairwise_prompt(sample: Dict[str, Any], rubric: str = PROMETHEUS_PAIRWISE_RUBRIC) -> str:
    instruction = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    response_a = normalize_text(sample.get("answer_a"))
    response_b = normalize_text(sample.get("answer_b"))
    reference = normalize_text(sample.get("reference"))
    if context:
        instruction = f"{instruction}\n\nContext:\n{context}"
    if not reference:
        reference = "No reference answer is provided."
    return "\n".join(
        [
            "###Task Description:",
            f"You are {DEFAULT_PROMETHEUS_JUDGE_IDENTITY}, an impartial evaluation judge for pairwise response ranking.",
            "Follow the rubric and decide whether Response A, Response B, or Tie is better.",
            "Start your answer with exactly one of: [RESULT] A, [RESULT] B, [RESULT] Tie.",
            "After the result marker, provide no more than three concise rationale sentences.",
            "",
            "###The instruction to evaluate:",
            instruction,
            "",
            "###Response A:",
            response_a,
            "",
            "###Response B:",
            response_b,
            "",
            "###Reference Answer:",
            reference,
            "",
            "###Score Rubric:",
            rubric,
            "",
            "###Feedback:",
        ]
    )


def build_prometheus_direct_prompt(sample: Dict[str, Any], rubric: str = PROMETHEUS_DIRECT_RUBRIC) -> str:
    instruction = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    response = normalize_text(sample.get("answer_a"))
    reference = normalize_text(sample.get("reference"))
    if context:
        instruction = f"{instruction}\n\nContext:\n{context}"
    if not reference:
        reference = "No reference answer is provided."
    return "\n".join(
        [
            "###Task Description:",
            f"You are {DEFAULT_PROMETHEUS_JUDGE_IDENTITY}, an impartial evaluation judge for direct response scoring.",
            "Return strict JSON when possible, with integer 1-5 scores for each dimension and overall_score.",
            "If JSON cannot be followed, end with exactly one final marker: [RESULT] 1, [RESULT] 2, [RESULT] 3, [RESULT] 4, or [RESULT] 5.",
            "",
            "###The instruction to evaluate:",
            instruction,
            "",
            "###Response:",
            response,
            "",
            "###Reference Answer:",
            reference,
            "",
            "###Score Rubric:",
            rubric,
            "",
            "###Output JSON Schema:",
            '{"relevance": 1, "completeness": 1, "factuality": 1, "instruction_following": 1, "clarity": 1, "safety": 1, "overall_score": 1}',
            "",
            "###Feedback:",
        ]
    )


def extract_prometheus_pairwise_label(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    raw = normalize_text(text)
    parsed: Dict[str, Any] = {"raw": raw}
    marker = re.search(r"\[RESULT\]\s*(A|B|Tie)\b", raw, flags=re.IGNORECASE)
    if marker:
        token = marker.group(1)
        normalized = "Tie" if token.lower() == "tie" else token.upper()
        parsed["result_token"] = normalized
        parsed["label"] = {"A": "A>B", "B": "B>A", "Tie": "Tie"}[normalized]
        return parsed["label"], parsed
    natural_label = extract_pairwise_label_from_natural_language(raw)
    if natural_label:
        parsed["label"] = natural_label
        parsed["parse_warning"] = "natural_language_result_marker_missing"
        return natural_label, parsed
    direct_label = re.search(r"\b(A>B|B>A|Tie)\b", raw, flags=re.IGNORECASE)
    if direct_label:
        label = direct_label.group(1)
        parsed["label"] = "Tie" if label.lower() == "tie" else label
        parsed["parse_warning"] = "missing_prometheus_result_marker"
        return parsed["label"], parsed
    return None, parsed


def extract_pairwise_label_from_natural_language(text: str) -> Optional[str]:
    raw = normalize_text(text).lower()
    a_conclusion_patterns = (
        r"\bresponse a (?:is|would be|should be)? ?(?:the )?(?:better|preferred|stronger|superior)(?: response)? than response b\b",
        r"\b(?:therefore|overall|in conclusion|based on the rubric),?.{0,80}\bresponse a (?:is|would be|should be)? ?(?:the )?(?:better|preferred|stronger|superior)\b",
    )
    b_conclusion_patterns = (
        r"\bresponse b (?:is|would be|should be)? ?(?:the )?(?:better|preferred|stronger|superior)(?: response)? than response a\b",
        r"\b(?:therefore|overall|in conclusion|based on the rubric),?.{0,80}\bresponse b (?:is|would be|should be)? ?(?:the )?(?:better|preferred|stronger|superior)\b",
    )
    conclusion_matches = [
        label
        for label, patterns in (("A>B", a_conclusion_patterns), ("B>A", b_conclusion_patterns))
        if any(re.search(pattern, raw) for pattern in patterns)
    ]
    if len(conclusion_matches) == 1:
        return conclusion_matches[0]

    tie_patterns = (
        r"\b(?:both|two) responses? (?:are|seem|appear) (?:equally good|equivalent|comparable|similar)",
        r"\b(?:it is|this is|the result is|should be) (?:a )?tie\b",
        r"\bneither response is (?:clearly )?better\b",
        r"\bboth responses? (?:are|were) identical(?: in content| in structure)?\b",
        r"\bboth responses? correctly\b",
        r"\bboth responses? (?:provide|present) (?:a )?(?:complete|accurate|similar|equivalent)\b",
    )
    a_positive_patterns = (
        r"\bresponse a (?:is|appears|seems|would be|should be) (?:the )?(?:better|best|preferred|stronger|superior)",
        r"\bresponse a (?:is|appears|seems) more (?:accurate|complete|comprehensive|detailed|thorough|specific|relevant|helpful|factual|factually correct|aligned)",
        r"\bresponse a provides a more (?:accurate|complete|comprehensive|detailed|thorough|specific|relevant|helpful|factual)",
        r"\bresponse a (?:correctly|directly|successfully|clearly) (?:identifies|answers|addresses|states|provides|presents|explains|calculates|follows|aligns)",
        r"\bresponse a (?:provides|presents|includes|offers|gives) (?:a |an )?(?:clear|correct|complete|comprehensive|relevant|practical|well-structured|accurate|factual)",
        r"\bresponse a (?:is|appears|seems) (?:complete|relevant|correct|accurate|factual|well-structured|clear)",
        r"\bresponse a (?:at least )?(?:attempts|makes an attempt)\b",
        r"\b(?:therefore|overall|in conclusion),? response a\b",
        r"\b(?:i would choose|choose|prefer|select) response a\b",
        r"\bresponse a (?:better|more effectively|more completely|more accurately)",
    )
    b_positive_patterns = (
        r"\bresponse b (?:is|appears|seems|would be|should be) (?:the )?(?:better|best|preferred|stronger|superior)",
        r"\bresponse b (?:is|appears|seems) more (?:accurate|complete|comprehensive|detailed|thorough|specific|relevant|helpful|factual|factually correct|aligned)",
        r"\bresponse b provides a more (?:accurate|complete|comprehensive|detailed|thorough|specific|relevant|helpful|factual)",
        r"\bresponse b (?:correctly|directly|successfully|clearly) (?:identifies|answers|addresses|states|provides|presents|explains|calculates|follows|aligns)",
        r"\bresponse b (?:provides|presents|includes|offers|gives) (?:a |an )?(?:clear|correct|complete|comprehensive|relevant|practical|well-structured|accurate|factual)",
        r"\bresponse b (?:is|appears|seems) (?:complete|relevant|correct|accurate|factual|well-structured|clear)",
        r"\bresponse b (?:at least )?(?:attempts|makes an attempt)\b",
        r"\b(?:therefore|overall|in conclusion),? response b\b",
        r"\b(?:i would choose|choose|prefer|select) response b\b",
        r"\bresponse b (?:better|more effectively|more completely|more accurately)",
    )
    a_negative_patterns = (
        r"\bresponse a (?:also )?(?:fails|does not|doesn't|lacks|incorrectly|only|merely|repeats)\b",
        r"\bresponse a (?:is|appears|seems) (?:incomplete|incorrect|less organized|less informative|harder to|not as)",
        r"\bresponse a (?:contains|has|includes) (?:several |multiple |many |some |significant |major )?(?:errors|mistakes|misconceptions|issues|flaws)\b",
        r"\bresponse a (?:is|appears|seems) (?:flawed|wrong|problematic|misleading)\b",
    )
    b_negative_patterns = (
        r"\bresponse b (?:also )?(?:fails|does not|doesn't|lacks|incorrectly|only|merely|repeats)\b",
        r"\bresponse b (?:is|appears|seems) (?:incomplete|incorrect|less organized|less informative|harder to|not as)",
        r"\bresponse b (?:contains|has|includes) (?:several |multiple |many |some |significant |major )?(?:errors|mistakes|misconceptions|issues|flaws)\b",
        r"\bresponse b (?:is|appears|seems) (?:flawed|wrong|problematic|misleading)\b",
    )
    has_tie = any(re.search(pattern, raw) for pattern in tie_patterns)
    has_a = any(re.search(pattern, raw) for pattern in a_positive_patterns) or any(
        re.search(pattern, raw) for pattern in b_negative_patterns
    )
    has_b = any(re.search(pattern, raw) for pattern in b_positive_patterns) or any(
        re.search(pattern, raw) for pattern in a_negative_patterns
    )
    matches = [label for label, present in (("Tie", has_tie), ("A>B", has_a), ("B>A", has_b)) if present]
    return matches[0] if len(matches) == 1 else None


def _coerce_1_to_5(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 <= score <= 5.0:
        return score
    return None


def extract_prometheus_direct_scores(text: str) -> Tuple[Optional[float], Dict[str, Any]]:
    raw = normalize_text(text)
    parsed: Dict[str, Any] = {"raw": raw, "dimensions": {}}
    json_match = re.search(r"\{.*\}", raw)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            for name in DIRECT_DIMENSIONS:
                score = _coerce_1_to_5(payload.get(name))
                if score is not None:
                    parsed["dimensions"][name] = score
            overall = _coerce_1_to_5(payload.get("overall_score"))
            if overall is not None:
                parsed["overall_score"] = overall
                return overall, parsed
        except json.JSONDecodeError:
            parsed["parse_warning"] = "invalid_json_object"

    for name in DIRECT_DIMENSIONS:
        match = re.search(rf"{name}\s*[:=]\s*([1-5](?:\.\d+)?)", raw, flags=re.IGNORECASE)
        if match:
            parsed["dimensions"][name] = float(match.group(1))
    overall_match = re.search(r"overall(?:_score|\s+score)?\s*[:=]\s*([1-5](?:\.\d+)?)", raw, flags=re.IGNORECASE)
    if overall_match:
        parsed["overall_score"] = float(overall_match.group(1))
        return parsed["overall_score"], parsed
    result_marker = re.search(r"\[RESULT\]\s*([1-5](?:\.\d+)?)\b", raw, flags=re.IGNORECASE)
    if result_marker:
        parsed["overall_score"] = float(result_marker.group(1))
        parsed["parse_warning"] = "result_marker_only_dimensions_missing"
        return parsed["overall_score"], parsed
    if parsed["dimensions"]:
        parsed["overall_score"] = round(mean(parsed["dimensions"].values()), 4)
        parsed["parse_warning"] = "overall_score_missing_mean_used"
        return parsed["overall_score"], parsed
    return None, parsed


class JudgeBackend:
    name = "base"

    def status(self) -> Dict[str, Any]:
        return {"available": True, "backend": self.name}

    def score_pairwise(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[str], Dict[str, Any]]:
        raise NotImplementedError

    def score_direct(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[float], Dict[str, Any]]:
        raise NotImplementedError


class HeuristicBackend(JudgeBackend):
    name = "heuristic"

    def score_pairwise(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[str], Dict[str, Any]]:
        return heuristic_pairwise_score(sample, config)

    def score_direct(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[float], Dict[str, Any]]:
        return heuristic_direct_score(sample, config)


class MockPrometheusBackend(JudgeBackend):
    name = "prometheus2"

    def __init__(self, output: str = "[RESULT] A") -> None:
        self.output = output

    def score_pairwise(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[str], Dict[str, Any]]:
        prompt = build_prometheus_pairwise_prompt(sample)
        label, parsed = extract_prometheus_pairwise_label(self.output)
        output = prometheus_output_payload(label, parsed, self.output, prompt, config, status="ok")
        return label, output

    def score_direct(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[float], Dict[str, Any]]:
        prompt = build_prometheus_direct_prompt(sample)
        score, parsed = extract_prometheus_direct_scores(self.output)
        output = prometheus_direct_output_payload(score, parsed, self.output, prompt, config, status="ok")
        return score, output


class PrometheusFamilyBackend(JudgeBackend):
    name = "prometheus_family"
    backbone = "Prometheus-family"

    def __init__(self, model_path: str, device: str = "auto") -> None:
        self.model_path = model_path
        self.device = device
        self._tokenizer = None
        self._model = None

    def status(self) -> Dict[str, Any]:
        missing = [
            package
            for package in ("torch", "transformers")
            if importlib.util.find_spec(package) is None
        ]
        return {
            "available": not missing and bool(self.model_path),
            "backend": self.name,
            "model_path": self.model_path,
            "backbone": self.backbone,
            "device": self.device,
            "missing_dependencies": missing,
            "python": platform.python_version(),
            "platform": platform.platform(),
        }

    def _load(self):
        status = self.status()
        if not status["available"]:
            raise RuntimeError(f"{self.backbone} backend unavailable: {status}")
        if self._tokenizer is None or self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                trust_remote_code=True,
            )
        return self._tokenizer, self._model

    def score_pairwise(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[str], Dict[str, Any]]:
        prompt = build_prometheus_pairwise_prompt(sample)
        tokenizer, model = self._load()
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=max(config.temperature, 1e-6),
            top_p=config.top_p,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        output_ids = generated[0][inputs["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        label, parsed = extract_prometheus_pairwise_label(raw_output)
        output = prometheus_output_payload(label, parsed, raw_output, prompt, config, status="ok")
        return label, output

    def score_direct(self, sample: Dict[str, Any], config: JudgeConfig) -> Tuple[Optional[float], Dict[str, Any]]:
        prompt = build_prometheus_direct_prompt(sample)
        tokenizer, model = self._load()
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generated = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0,
            temperature=max(config.temperature, 1e-6),
            top_p=config.top_p,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        output_ids = generated[0][inputs["input_ids"].shape[-1] :]
        raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
        score, parsed = extract_prometheus_direct_scores(raw_output)
        output = prometheus_direct_output_payload(score, parsed, raw_output, prompt, config, status="ok")
        return score, output


class Prometheus2Backend(PrometheusFamilyBackend):
    name = "prometheus2"
    backbone = "Prometheus 2"


class MPrometheus3BBackend(PrometheusFamilyBackend):
    name = "m_prometheus"
    backbone = "M-Prometheus-3B"


def pairwise_prompt_template_name(config: JudgeConfig) -> str:
    if config.backend == "m_prometheus":
        return "m_prometheus_pairwise_v1"
    return "prometheus2_pairwise_v1"


def direct_prompt_template_name(config: JudgeConfig) -> str:
    if config.backend == "m_prometheus":
        return "m_prometheus_direct_v1"
    return "prometheus2_direct_v1"


def is_prometheus_family_backend(config: JudgeConfig) -> bool:
    return config.backend in {"prometheus2", "m_prometheus"}


def prometheus_output_payload(
    label: Optional[str],
    parsed: Dict[str, Any],
    raw_output: str,
    prompt: str,
    config: JudgeConfig,
    status: str,
) -> Dict[str, Any]:
    score_a, score_b = {
        "A>B": (1.0, 0.0),
        "B>A": (0.0, 1.0),
        "Tie": (0.5, 0.5),
    }.get(label or "", (None, None))
    return {
        "judge_name": config.name,
        "judge_version": config.version,
        "judge_backend": config.backend,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "mode": config.mode,
        "prompt_template": pairwise_prompt_template_name(config),
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_label": label,
        "parsed_scores": {"score_a": score_a, "score_b": score_b},
        "parse_status": status if label else "failed",
        "parse_metadata": parsed,
    }


def prometheus_direct_output_payload(
    score: Optional[float],
    parsed: Dict[str, Any],
    raw_output: str,
    prompt: str,
    config: JudgeConfig,
    status: str,
) -> Dict[str, Any]:
    dimensions = parsed.get("dimensions", {})
    return {
        "judge_name": config.name,
        "judge_version": config.version,
        "judge_backend": config.backend,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "mode": config.mode,
        "prompt_template": direct_prompt_template_name(config),
        "prompt": prompt,
        "raw_output": raw_output,
        "parsed_label": None,
        "parsed_scores": {"overall_score": score, **dimensions},
        "parse_status": status if score is not None else "failed",
        "parse_metadata": parsed,
    }


def make_backend(config: JudgeConfig) -> JudgeBackend:
    if config.backend == "prometheus2":
        if config.model_path == "mock":
            return MockPrometheusBackend("[RESULT] A")
        return Prometheus2Backend(config.model_path, device=config.device)
    if config.backend == "m_prometheus":
        if config.model_path == "mock":
            return MockPrometheusBackend("[RESULT] A")
        return MPrometheus3BBackend(config.model_path, device=config.device)
    return HeuristicBackend()


def heuristic_pairwise_score(sample: Dict[str, Any], config: JudgeConfig) -> Tuple[str, Dict[str, Any]]:
    prompt = normalize_text(sample.get("prompt"))
    answer_a = normalize_text(sample.get("answer_a"))
    answer_b = normalize_text(sample.get("answer_b"))
    context = normalize_text(sample.get("context"))

    a_len = len(answer_a)
    b_len = len(answer_b)
    a_sentence_count = max(1, len([s for s in re.split(r"(?<=[.!?。！？])\s+", answer_a) if s.strip()]))
    b_sentence_count = max(1, len([s for s in re.split(r"(?<=[.!?。！？])\s+", answer_b) if s.strip()]))
    prompt_len = len(prompt)
    context_len = len(context)

    # Baseline judge: a transparent heuristic that combines length, structure,
    # prompt-context overlap, and a simple tie band. It is intentionally simple
    # so downstream calibration and perturbation analysis can be layered on top.
    a_score = 0.0
    b_score = 0.0

    if a_len >= 12:
        a_score += 0.2
    if b_len >= 12:
        b_score += 0.2
    a_score += min(0.5, a_sentence_count * 0.05)
    b_score += min(0.5, b_sentence_count * 0.05)

    if a_len > b_len:
        a_score += 0.15
    elif b_len > a_len:
        b_score += 0.15

    prompt_terms = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{3,}", prompt.lower()))
    context_terms = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{3,}", context.lower()))
    if prompt_terms and context_terms:
        overlap = len(prompt_terms & context_terms) / max(1, len(prompt_terms))
        a_score += overlap * 0.1
        b_score += overlap * 0.1

    if prompt_len > 0 and context_len > 0:
        balance = 1.0 - min(1.0, abs(prompt_len - context_len) / max(prompt_len, context_len))
        a_score += balance * 0.05
        b_score += balance * 0.05

    diff = a_score - b_score
    if abs(diff) < 0.08:
        label = "Tie"
    elif diff > 0:
        label = "A>B"
    else:
        label = "B>A"

    output = {
        "judge_name": config.name,
        "judge_version": config.version,
        "judge_backend": "heuristic",
        "temperature": config.temperature,
        "top_p": config.top_p,
        "mode": config.mode,
        "prompt_template": config.prompt_template,
        "raw_output": f"Predicted label: {label}; score_a={a_score:.3f}; score_b={b_score:.3f}",
        "parsed_label": label,
        "parsed_scores": {"score_a": round(a_score, 4), "score_b": round(b_score, 4)},
        "parse_status": "ok",
    }
    return label, output


def heuristic_direct_score(sample: Dict[str, Any], config: JudgeConfig) -> Tuple[float, Dict[str, Any]]:
    answer = normalize_text(sample.get("answer_a"))
    prompt = normalize_text(sample.get("prompt"))
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference"))
    length_score = min(5.0, max(1.0, len(answer) / 120.0 + 1.0))
    prompt_overlap = len(token_set(prompt) & token_set(answer)) / max(1, len(token_set(prompt))) if prompt else 0.0
    context_overlap = len(token_set(context) & token_set(answer)) / max(1, len(token_set(context))) if context else 0.0
    reference_overlap = len(token_set(reference) & token_set(answer)) / max(1, len(token_set(reference))) if reference else 0.0
    dimensions = {
        "relevance": round(1.0 + 4.0 * min(1.0, prompt_overlap + 0.2), 4),
        "completeness": round(length_score, 4),
        "factuality": round(1.0 + 4.0 * max(context_overlap, reference_overlap, 0.25), 4),
        "instruction_following": round(1.0 + 4.0 * min(1.0, prompt_overlap + 0.15), 4),
        "clarity": round(min(5.0, 2.5 + sentence_clarity_bonus(answer)), 4),
        "safety": 4.0,
    }
    overall = round(mean(dimensions.values()), 4)
    output = {
        "judge_name": config.name,
        "judge_version": config.version,
        "judge_backend": "heuristic",
        "temperature": config.temperature,
        "top_p": config.top_p,
        "mode": config.mode,
        "prompt_template": config.prompt_template,
        "raw_output": json.dumps({"overall_score": overall, **dimensions}, ensure_ascii=False),
        "parsed_label": None,
        "parsed_scores": {"overall_score": overall, **dimensions},
        "parse_status": "ok",
    }
    return overall, output


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]{2,}", normalize_text(text).lower()))


def sentence_clarity_bonus(text: str) -> float:
    sentences = [s for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    return min(2.0, len(sentences) * 0.25)


def evaluate_samples(
    samples: Sequence[Dict[str, Any]],
    config: JudgeConfig,
    backend: Optional[JudgeBackend] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if backend is None:
        backend = make_backend(config)
    if config.mode == "direct":
        return evaluate_direct_samples_with_backend(samples, config, backend)
    return evaluate_samples_with_backend(samples, config, backend)


def evaluate_samples_with_backend(
    samples: Sequence[Dict[str, Any]],
    config: JudgeConfig,
    backend: JudgeBackend,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, Any]] = []
    backend_status = backend.status()
    fallback_reason: Optional[str] = None

    for sample in samples:
        try:
            if is_prometheus_family_backend(config) and config.allow_fallback and fallback_reason:
                pred_label, raw = fallback_pairwise_raw(sample, config, backend_status, error=fallback_reason)
            elif is_prometheus_family_backend(config) and not backend_status.get("available", False):
                if not config.allow_fallback:
                    raise RuntimeError(f"{config.backend} backend unavailable: {backend_status}")
                pred_label, raw = fallback_pairwise_raw(sample, config, backend_status)
            else:
                pred_label, raw = backend.score_pairwise(sample, config)
        except Exception as exc:
            if is_prometheus_family_backend(config) and config.allow_fallback:
                fallback_reason = str(exc)
                pred_label, raw = fallback_pairwise_raw(sample, config, backend_status, error=fallback_reason)
            else:
                pred_label = None
                raw = {
                    "judge_name": config.name,
                    "judge_version": config.version,
                    "judge_backend": backend.name,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "mode": config.mode,
                    "prompt_template": config.prompt_template,
                    "raw_output": "",
                    "parsed_label": None,
                    "parsed_scores": {"score_a": None, "score_b": None},
                    "parse_status": "backend_error",
                    "backend_status": backend_status,
                    "error": str(exc),
                }
        gold_label = sample.get("human_label")
        gold_score = score_from_token(str(gold_label)) if gold_label else None
        pred_score = score_from_token(pred_label or "")
        if pred_score is None:
            parse_failures.append(
                {
                    "id": sample.get("id"),
                    "dataset": sample.get("dataset"),
                    "task_type": sample.get("task_type"),
                    "raw_output": raw["raw_output"],
                    "parse_status": raw.get("parse_status"),
                    "backend_status": raw.get("backend_status", backend_status),
                    "error": raw.get("error"),
                }
            )
            continue
        results.append(
            {
                "id": sample.get("id"),
                "dataset": sample.get("dataset"),
                "task_type": sample.get("task_type"),
                "gold_label": gold_label,
                "pred_label": pred_label,
                "gold_score": gold_score,
                "pred_score": pred_score,
                "prompt": sample.get("prompt"),
                "context": sample.get("context"),
                "answer_a": sample.get("answer_a"),
                "answer_b": sample.get("answer_b"),
                "raw_output": raw["raw_output"],
                "judge_backend": raw.get("judge_backend", backend.name),
                "judge_name": raw.get("judge_name", config.name),
                "judge_version": raw.get("judge_version", config.version),
                "prompt_template": raw.get("prompt_template", config.prompt_template),
                "top_p": raw.get("top_p", config.top_p),
                "parsed_scores": raw["parsed_scores"],
                "parse_status": raw["parse_status"],
                "backend_status": raw.get("backend_status", backend_status),
                "parse_metadata": raw.get("parse_metadata", {}),
                "fallback_from": raw.get("fallback_from"),
                "fallback_reason": raw.get("fallback_reason"),
            }
        )

    return summarize_results(results, parse_failures, config, backend_status), results


def fallback_pairwise_raw(
    sample: Dict[str, Any],
    config: JudgeConfig,
    backend_status: Dict[str, Any],
    error: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    pred_label, raw = heuristic_pairwise_score(sample, config)
    raw["judge_backend"] = "heuristic_fallback"
    raw["backend_status"] = backend_status
    raw["prompt_template"] = pairwise_prompt_template_name(config)
    raw["fallback_from"] = config.backend
    raw["parse_status"] = "fallback_ok"
    if error:
        raw["fallback_reason"] = error
    return pred_label, raw


def fallback_direct_raw(
    sample: Dict[str, Any],
    config: JudgeConfig,
    backend_status: Dict[str, Any],
    error: Optional[str] = None,
) -> Tuple[float, Dict[str, Any]]:
    pred_score, raw = heuristic_direct_score(sample, config)
    raw["judge_backend"] = "heuristic_fallback"
    raw["backend_status"] = backend_status
    raw["prompt_template"] = direct_prompt_template_name(config)
    raw["fallback_from"] = config.backend
    raw["parse_status"] = "fallback_ok"
    if error:
        raw["fallback_reason"] = error
    return pred_score, raw


def evaluate_direct_samples_with_backend(
    samples: Sequence[Dict[str, Any]],
    config: JudgeConfig,
    backend: JudgeBackend,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    results: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, Any]] = []
    backend_status = backend.status()
    fallback_reason: Optional[str] = None

    for sample in samples:
        try:
            if is_prometheus_family_backend(config) and config.allow_fallback and fallback_reason:
                pred_score, raw = fallback_direct_raw(sample, config, backend_status, error=fallback_reason)
            elif is_prometheus_family_backend(config) and not backend_status.get("available", False):
                if not config.allow_fallback:
                    raise RuntimeError(f"{config.backend} backend unavailable: {backend_status}")
                pred_score, raw = fallback_direct_raw(sample, config, backend_status)
            else:
                pred_score, raw = backend.score_direct(sample, config)
        except Exception as exc:
            if is_prometheus_family_backend(config) and config.allow_fallback:
                fallback_reason = str(exc)
                pred_score, raw = fallback_direct_raw(sample, config, backend_status, error=fallback_reason)
            else:
                pred_score = None
                raw = {
                    "judge_name": config.name,
                    "judge_version": config.version,
                    "judge_backend": backend.name,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "mode": config.mode,
                    "prompt_template": config.prompt_template,
                    "raw_output": "",
                    "parsed_label": None,
                    "parsed_scores": {"overall_score": None},
                    "parse_status": "backend_error",
                    "backend_status": backend_status,
                    "error": str(exc),
                }
        if pred_score is None:
            parse_failures.append(
                {
                    "id": sample.get("id"),
                    "dataset": sample.get("dataset"),
                    "task_type": sample.get("task_type"),
                    "raw_output": raw["raw_output"],
                    "parse_status": raw.get("parse_status"),
                    "backend_status": raw.get("backend_status", backend_status),
                    "error": raw.get("error"),
                }
            )
            continue
        results.append(
            {
                "id": sample.get("id"),
                "dataset": sample.get("dataset"),
                "task_type": sample.get("task_type"),
                "gold_label": sample.get("human_label"),
                "pred_label": None,
                "gold_score": None,
                "pred_score": pred_score,
                "prompt": sample.get("prompt"),
                "context": sample.get("context"),
                "answer_a": sample.get("answer_a"),
                "answer_b": sample.get("answer_b"),
                "raw_output": raw["raw_output"],
                "judge_backend": raw.get("judge_backend", backend.name),
                "judge_name": raw.get("judge_name", config.name),
                "judge_version": raw.get("judge_version", config.version),
                "prompt_template": raw.get("prompt_template", config.prompt_template),
                "top_p": raw.get("top_p", config.top_p),
                "parsed_scores": raw["parsed_scores"],
                "parse_status": raw["parse_status"],
                "backend_status": raw.get("backend_status", backend_status),
                "parse_metadata": raw.get("parse_metadata", {}),
                "fallback_from": raw.get("fallback_from"),
                "fallback_reason": raw.get("fallback_reason"),
            }
        )

    return summarize_results(results, parse_failures, config, backend_status), results


def pairwise_accuracy(rows: Sequence[Dict[str, Any]]) -> float:
    pair_rows = [row for row in rows if row.get("gold_label") in PAIRWISE_LABELS and row.get("pred_label") in PAIRWISE_LABELS]
    if not pair_rows:
        return 0.0
    correct = sum(1 for row in pair_rows if row["gold_label"] == row["pred_label"])
    return round(correct / len(pair_rows), 4)


def tie_rate(rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    ties = sum(1 for row in rows if row["pred_label"] == "Tie")
    return round(ties / len(rows), 4)


def summarize_results(
    rows: Sequence[Dict[str, Any]],
    parse_failures: Sequence[Dict[str, Any]],
    config: JudgeConfig,
    backend_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    by_task = defaultdict(list)
    by_dataset = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
        by_dataset[row["dataset"]].append(row)

    overall = {
        "pairwise_accuracy": pairwise_accuracy(rows) if config.mode == "pairwise" else None,
        "tie_rate": tie_rate(rows) if config.mode == "pairwise" else None,
        "pred_label_distribution": dict(Counter(row["pred_label"] for row in rows)) if config.mode == "pairwise" else None,
        "gold_label_distribution": dict(Counter(str(row["gold_label"]) for row in rows if row.get("gold_label"))),
        "direct_score_mean": round(mean(row["pred_score"] for row in rows), 4) if config.mode == "direct" and rows else None,
        "direct_score_distribution": dict(Counter(round(float(row["pred_score"])) for row in rows))
        if config.mode == "direct" and rows
        else None,
    }
    summary = {
        "created_at": utc_now(),
        "judge": {
            "name": config.name,
            "version": config.version,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "mode": config.mode,
            "prompt_template": config.prompt_template,
            "backend": config.backend,
            "model_path": config.model_path,
            "max_new_tokens": config.max_new_tokens,
            "device": config.device,
        },
        "backend_status": backend_status or {},
        "sample_count": len(rows),
        "parse_failure_count": len(parse_failures),
        "overall": overall,
        "by_task_type": {
            task: {
                "count": len(items),
                "pairwise_accuracy": pairwise_accuracy(items) if config.mode == "pairwise" else None,
                "tie_rate": tie_rate(items) if config.mode == "pairwise" else None,
                "pred_label_distribution": dict(Counter(row["pred_label"] for row in items))
                if config.mode == "pairwise"
                else None,
                "direct_score_mean": round(mean(row["pred_score"] for row in items), 4)
                if config.mode == "direct" and items
                else None,
            }
            for task, items in by_task.items()
        },
        "by_dataset": {
            dataset: {
                "count": len(items),
                "pairwise_accuracy": pairwise_accuracy(items) if config.mode == "pairwise" else None,
                "tie_rate": tie_rate(items) if config.mode == "pairwise" else None,
                "direct_score_mean": round(mean(row["pred_score"] for row in items), 4)
                if config.mode == "direct" and items
                else None,
            }
            for dataset, items in by_dataset.items()
        },
        "parse_failures": parse_failures,
    }
    return summary


def load_default_samples() -> List[Dict[str, Any]]:
    path = PROCESSED / "bea_judge_core_2400.json"
    payload = read_dataset(path)
    return payload["samples"]


def pairwise_samples(samples: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        sample
        for sample in samples
        if sample.get("human_label") in PAIRWISE_LABELS and normalize_text(sample.get("answer_b"))
    ]


def base_score_rows_for_disk(
    rows: Sequence[Dict[str, Any]],
    parse_failures: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    output_rows = list(rows)
    for failure in parse_failures:
        backend_status = failure.get("backend_status") or {}
        output_rows.append(
            {
                "id": failure.get("id"),
                "dataset": failure.get("dataset"),
                "task_type": failure.get("task_type"),
                "gold_label": failure.get("gold_label"),
                "pred_label": None,
                "gold_score": failure.get("gold_score"),
                "pred_score": None,
                "raw_output": failure.get("raw_output", ""),
                "judge_backend": backend_status.get("backend", "backend_error"),
                "parsed_scores": {"score_a": None, "score_b": None},
                "parse_status": failure.get("parse_status", "backend_error"),
                "backend_status": backend_status,
                "error": failure.get("error"),
            }
        )
    return output_rows


def is_valid_pairwise_score_row(row: Dict[str, Any]) -> bool:
    scores = row.get("parsed_scores") or {}
    return (
        row.get("pred_label") in PAIRWISE_LABELS
        and scores.get("score_a") is not None
        and scores.get("score_b") is not None
        and row.get("parse_status") not in {"failed", "backend_error"}
    )


def split_base_score_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    valid_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for row in rows:
        if is_valid_pairwise_score_row(row):
            valid_rows.append(row)
        else:
            failures.append(
                {
                    "id": row.get("id"),
                    "dataset": row.get("dataset"),
                    "task_type": row.get("task_type"),
                    "gold_label": row.get("gold_label"),
                    "gold_score": row.get("gold_score"),
                    "raw_output": row.get("raw_output", ""),
                    "parse_status": row.get("parse_status", "failed"),
                    "backend_status": row.get("backend_status", {}),
                    "error": row.get("error"),
                }
            )
    return valid_rows, failures


def read_score_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"score rows must be a list: {path}")
    return payload


def existing_score_rows(run_dir: Path) -> List[Dict[str, Any]]:
    partial_path = run_dir / "base_scores.partial.json"
    final_path = run_dir / "base_scores.json"
    if partial_path.exists():
        return read_score_rows(partial_path)
    return read_score_rows(final_path)


def write_checkpoint_files(
    run_dir: Path,
    summary: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    *,
    final: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    score_name = "base_scores.json" if final else "base_scores.partial.json"
    summary_name = "summary.json" if final else "summary.partial.json"
    (run_dir / score_name).write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "parse_failures.json").write_text(
        json.dumps(summary["parse_failures"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_resumable_evaluation(
    samples: Sequence[Dict[str, Any]],
    config: JudgeConfig,
    run_dir: Path,
    *,
    checkpoint_interval: int = 25,
    backend: Optional[JudgeBackend] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if backend is None:
        backend = make_backend(config)
    backend_status = backend.status()
    all_rows = existing_score_rows(run_dir)
    seen_ids = {str(row.get("id")) for row in all_rows if row.get("id") is not None}
    remaining = [sample for sample in samples if str(sample.get("id")) not in seen_ids]
    evaluated_new_count = 0
    interval = max(1, checkpoint_interval)

    for sample in remaining:
        _summary, new_valid_rows = evaluate_samples([sample], config, backend=backend)
        all_rows = base_score_rows_for_disk(
            split_base_score_rows(all_rows)[0] + new_valid_rows,
            split_base_score_rows(all_rows)[1] + _summary["parse_failures"],
        )
        evaluated_new_count += 1
        if evaluated_new_count % interval == 0:
            valid_rows, failures = split_base_score_rows(all_rows)
            checkpoint_summary = summarize_results(valid_rows, failures, config, backend_status)
            checkpoint_summary["checkpoint"] = {
                "run_dir": str(run_dir),
                "skipped_existing_count": len(seen_ids),
                "evaluated_new_count": evaluated_new_count,
                "remaining_count": len(remaining) - evaluated_new_count,
                "checkpoint_interval": interval,
                "final": False,
            }
            write_checkpoint_files(run_dir, checkpoint_summary, all_rows, final=False)

    valid_rows, failures = split_base_score_rows(all_rows)
    summary = summarize_results(valid_rows, failures, config, backend_status)
    summary["checkpoint"] = {
        "run_dir": str(run_dir),
        "skipped_existing_count": len(seen_ids),
        "evaluated_new_count": evaluated_new_count,
        "remaining_count": 0,
        "checkpoint_interval": interval,
        "final": True,
    }
    summary["coverage"] = {
        "total_pairwise_samples": len(samples),
        "parsed_rows": len(valid_rows),
        "failed_rows": sum(1 for row in all_rows if row.get("parse_status") == "failed"),
        "backend_error_rows": sum(1 for row in all_rows if row.get("parse_status") == "backend_error"),
        "parse_success_rate": round(len(valid_rows) / len(samples), 4) if samples else 0.0,
    }
    write_checkpoint_files(run_dir, summary, all_rows, final=True)
    return summary, all_rows


def write_outputs(summary: Dict[str, Any], rows: Sequence[Dict[str, Any]], config: JudgeConfig) -> None:
    ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = EVAL_OUT / f"{config.name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "base_scores.json").write_text(
        json.dumps(
            base_score_rows_for_disk(rows, summary["parse_failures"]),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "parse_failures.json").write_text(
        json.dumps(summary["parse_failures"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latest = EVAL_OUT / "latest_summary.json"
    latest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the baseline BEA-Judge scoring module.")
    parser.add_argument("--input", type=str, default=str(PROCESSED / "bea_judge_core_2400.json"))
    parser.add_argument("--config", type=str, default=str(DEFAULT_JUDGE_CONFIG_PATH))
    parser.add_argument("--mode", type=str, default=None, choices=["pairwise", "direct"])
    parser.add_argument("--judge-name", type=str, default=None)
    parser.add_argument("--judge-version", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--prompt-template", type=str, default=None)
    parser.add_argument("--backend", type=str, default=None, choices=["heuristic", "prometheus2", "m_prometheus"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--pairwise-only", action="store_true")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = load_judge_config(Path(args.config))
    config = JudgeConfig(
        name=args.judge_name or defaults.name,
        version=args.judge_version or defaults.version,
        temperature=defaults.temperature if args.temperature is None else args.temperature,
        top_p=defaults.top_p if args.top_p is None else args.top_p,
        mode=args.mode or defaults.mode,
        prompt_template=args.prompt_template or defaults.prompt_template,
        backend=args.backend or defaults.backend,
        model_path=args.model_path if args.model_path is not None else defaults.model_path,
        max_new_tokens=defaults.max_new_tokens if args.max_new_tokens is None else args.max_new_tokens,
        device=args.device or defaults.device,
        allow_fallback=args.allow_fallback,
    )
    payload = read_dataset(Path(args.input))
    samples = payload["samples"]
    if args.pairwise_only:
        samples = pairwise_samples(samples)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]
    if args.resume and not args.run_dir:
        raise ValueError("--resume requires --run-dir")
    if args.run_dir:
        summary, _rows = run_resumable_evaluation(
            samples,
            config,
            Path(args.run_dir),
            checkpoint_interval=args.checkpoint_interval,
        )
    else:
        summary, rows = evaluate_samples(samples, config)
        write_outputs(summary, rows, config)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["by_task_type"], ensure_ascii=False, indent=2))
    if "coverage" in summary:
        print(json.dumps(summary["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
