"""Evidence-enhanced factuality features for BEA-Judge."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


PAIRWISE_FACTUALITY_LABELS = {"A>B", "B>A", "Tie"}
SINGLE_FACTUALITY_LABELS = {"supported", "unsupported", "ambiguous"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}
NEGATION_PATTERNS = (
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "without",
    "cannot",
    "can't",
    "won't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "doesn't",
    "didn't",
    "不",
    "未",
    "没有",
    "无",
)
COMPARATIVE_GROUPS = {
    "increase": {
        "more",
        "greater",
        "higher",
        "larger",
        "longer",
        "better",
        "increased",
        "increases",
        "increase",
        "exceeds",
        "above",
        "over",
        "最多",
        "更高",
        "更大",
    },
    "decrease": {
        "less",
        "fewer",
        "lower",
        "smaller",
        "shorter",
        "worse",
        "decreased",
        "decreases",
        "decrease",
        "below",
        "under",
        "最少",
        "更低",
        "更小",
    },
    "maximum": {"most", "largest", "highest", "greatest", "best", "maximum", "max", "top"},
    "minimum": {"least", "smallest", "lowest", "worst", "minimum", "min", "bottom"},
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clip01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, value))


def safe_round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(clip01(float(value)), 6)


def _tokens(text: Any) -> List[str]:
    normalized = normalize_text(text).lower()
    raw_tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    tokens = [token for token in raw_tokens if token not in STOPWORDS]
    return tokens


def _token_set(text: Any) -> set[str]:
    return set(_tokens(text))


def _numbers(text: Any) -> set[str]:
    values = set()
    for match in re.findall(r"\d+(?:[,.]\d+)?", normalize_text(text)):
        values.add(match.replace(",", ""))
    return values


def _dates(text: Any) -> set[str]:
    normalized = normalize_text(text)
    values = set(re.findall(r"\b(?:1[6-9]\d{2}|20\d{2}|21\d{2})\b", normalized))
    values.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", normalized))
    month_names = (
        "january|february|march|april|may|june|july|august|september|october|november|december|"
        "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    )
    for match in re.findall(rf"\b(?:{month_names})\.?\s+\d{{1,2}}(?:,\s*\d{{4}})?\b", normalized, flags=re.IGNORECASE):
        values.add(match.lower())
    return values


def _entity_like_tokens(text: Any) -> set[str]:
    raw = normalize_text(text)
    values = set()
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", raw):
        lower = token.lower()
        if lower in STOPWORDS:
            continue
        if re.search(r"\d", token) and re.search(r"[A-Za-z]", token):
            values.add(lower)
        elif re.fullmatch(r"[A-Z]{2,}", token):
            values.add(lower)
        elif re.fullmatch(r"[A-Z][a-z]{2,}(?:-[A-Z][a-z]{2,})?", token):
            values.add(lower)
        elif len(token) >= 9:
            values.add(lower)
    return values


def _alias_normalized_entity_tokens(text: Any) -> set[str]:
    raw = normalize_text(text)
    values = set()
    for token in _entity_like_tokens(raw):
        normalized = token.replace(".", "").replace("-", "")
        if normalized.endswith("'s"):
            normalized = normalized[:-2]
        if len(normalized) > 4 and normalized.endswith("s"):
            values.add(normalized[:-1])
        values.add(normalized)

    capitalized = re.findall(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5}\b", raw)
    for phrase in capitalized:
        initials = "".join(part[0].lower() for part in phrase.split() if part and part.lower() not in STOPWORDS)
        if len(initials) >= 2:
            values.add(initials)
    return {value for value in values if value}


def _gap_ratio(answer_values: set[str], *evidence_blocks: Any) -> float:
    if not answer_values:
        return 0.0
    evidence_values = set()
    for block in evidence_blocks:
        evidence_values.update(block)
    missing = answer_values - evidence_values
    return clip01(len(missing) / len(answer_values))


def _coverage(answer: Any, evidence: Any) -> float:
    answer_tokens = _token_set(answer)
    evidence_tokens = _token_set(evidence)
    if not answer_tokens or not evidence_tokens:
        return 0.0
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def _long_token_coverage(answer: Any, evidence: Any) -> float:
    answer_tokens = {
        token
        for token in _tokens(answer)
        if token.isdigit() or len(token) >= 4 or re.fullmatch(r"[\u4e00-\u9fff]", token)
    }
    evidence_tokens = _token_set(evidence)
    if not answer_tokens or not evidence_tokens:
        return 0.0
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def _support_score(answer: Any, evidence: Any) -> float:
    answer_text = normalize_text(answer)
    evidence_text = normalize_text(evidence)
    if not answer_text or not evidence_text:
        return 0.0
    token_support = _coverage(answer_text, evidence_text)
    long_token_support = _long_token_coverage(answer_text, evidence_text)
    answer_numbers = _numbers(answer_text)
    evidence_numbers = _numbers(evidence_text)
    if answer_numbers:
        numeric_support = len(answer_numbers & evidence_numbers) / len(answer_numbers)
        return clip01(0.45 * token_support + 0.30 * long_token_support + 0.25 * numeric_support)
    return clip01(0.65 * token_support + 0.35 * long_token_support)


def _numeric_gap(answer: Any, *evidence_blocks: Any) -> float:
    answer_numbers = _numbers(answer)
    if not answer_numbers:
        return 0.0
    evidence_numbers = set()
    for block in evidence_blocks:
        evidence_numbers.update(_numbers(block))
    missing = answer_numbers - evidence_numbers
    return clip01(len(missing) / len(answer_numbers))


def _date_gap(answer: Any, *evidence_blocks: Any) -> float:
    return _gap_ratio(_dates(answer), *(_dates(block) for block in evidence_blocks))


def _entity_gap(answer: Any, *evidence_blocks: Any) -> float:
    return _gap_ratio(_entity_like_tokens(answer), *(_entity_like_tokens(block) for block in evidence_blocks))


def _entity_alias_gap(answer: Any, *evidence_blocks: Any) -> float:
    return _gap_ratio(
        _alias_normalized_entity_tokens(answer),
        *(_alias_normalized_entity_tokens(block) for block in evidence_blocks),
    )


def _contains_negation(text: Any) -> bool:
    normalized = normalize_text(text).lower()
    return any(re.search(rf"(^|\W){re.escape(pattern)}($|\W)", normalized) for pattern in NEGATION_PATTERNS)


def _negation_mismatch(answer: Any, evidence: Any) -> float:
    answer_sentences = _sentences(answer)
    evidence_sentences = _sentences(evidence)
    if not answer_sentences or not evidence_sentences:
        answer_has_negation = _contains_negation(answer)
        evidence_has_negation = _contains_negation(evidence)
        return 1.0 if answer_has_negation != evidence_has_negation and _coverage(answer, evidence) >= 0.15 else 0.0
    mismatches = 0
    compared = 0
    for answer_sentence in answer_sentences:
        best_sentence = max(evidence_sentences, key=lambda evidence_sentence: _support_score(answer_sentence, evidence_sentence))
        if _support_score(answer_sentence, best_sentence) < 0.20:
            continue
        compared += 1
        if _contains_negation(answer_sentence) != _contains_negation(best_sentence):
            mismatches += 1
    return clip01(mismatches / compared) if compared else 0.0


def _comparative_groups(text: Any) -> set[str]:
    normalized_tokens = set(_tokens(text))
    groups = set()
    for group, terms in COMPARATIVE_GROUPS.items():
        if normalized_tokens & terms:
            groups.add(group)
    return groups


def _comparative_mismatch(answer: Any, evidence: Any) -> float:
    answer_sentences = _sentences(answer)
    evidence_sentences = _sentences(evidence)
    if not answer_sentences or not evidence_sentences:
        answer_groups = _comparative_groups(answer)
        if not answer_groups:
            return 0.0
        return clip01(len(answer_groups - _comparative_groups(evidence)) / len(answer_groups))
    scores = []
    for answer_sentence in answer_sentences:
        answer_groups = _comparative_groups(answer_sentence)
        if not answer_groups:
            continue
        best_sentence = max(evidence_sentences, key=lambda evidence_sentence: _support_score(answer_sentence, evidence_sentence))
        if _support_score(answer_sentence, best_sentence) < 0.20:
            continue
        missing = answer_groups - _comparative_groups(best_sentence)
        scores.append(len(missing) / len(answer_groups))
    return clip01(sum(scores) / len(scores)) if scores else 0.0


def _sentences(text: Any) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+|[\n\r]+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _sentence_support_stats(answer: Any, evidence: Any) -> Dict[str, float]:
    answer_sentences = _sentences(answer)
    evidence_sentences = _sentences(evidence)
    if not answer_sentences or not evidence_sentences:
        return {
            "sentence_max_support": 0.0,
            "sentence_mean_support": 0.0,
            "low_support_sentence_ratio": 1.0 if answer_sentences else 0.0,
        }
    sentence_scores = []
    for answer_sentence in answer_sentences:
        best = max(_support_score(answer_sentence, evidence_sentence) for evidence_sentence in evidence_sentences)
        sentence_scores.append(best)
    low_support = [score for score in sentence_scores if score < 0.35]
    return {
        "sentence_max_support": clip01(max(sentence_scores)),
        "sentence_mean_support": clip01(sum(sentence_scores) / len(sentence_scores)),
        "low_support_sentence_ratio": clip01(len(low_support) / len(sentence_scores)),
    }


def _anchored_sentence_gap_stats(answer: Any, evidence: Any) -> Dict[str, float]:
    answer_sentences = _sentences(answer)
    if not answer_sentences:
        return {
            "low_support_anchor_sentence_ratio": 0.0,
            "max_low_support_anchor_gap": 0.0,
            "anchored_hallucination_severity": 0.0,
        }
    evidence_sentences = _sentences(evidence)
    low_support_anchor_count = 0
    max_gap = 0.0
    severity_scores: List[float] = []
    for answer_sentence in answer_sentences:
        if evidence_sentences:
            support = max(_support_score(answer_sentence, evidence_sentence) for evidence_sentence in evidence_sentences)
        else:
            support = _support_score(answer_sentence, evidence)
        anchor_gap = max(
            _numeric_gap(answer_sentence, evidence),
            _date_gap(answer_sentence, evidence),
            _entity_gap(answer_sentence, evidence),
            _entity_alias_gap(answer_sentence, evidence),
        )
        if support < 0.35 and anchor_gap > 0.0:
            low_support_anchor_count += 1
        if support < 0.50:
            max_gap = max(max_gap, anchor_gap)
        severity_scores.append((1.0 - support) * anchor_gap)
    return {
        "low_support_anchor_sentence_ratio": clip01(low_support_anchor_count / len(answer_sentences)),
        "max_low_support_anchor_gap": clip01(max_gap),
        "anchored_hallucination_severity": clip01(sum(severity_scores) / len(severity_scores)),
    }


def _task_form(sample: Dict[str, Any]) -> str:
    meta = sample.get("metadata") or {}
    explicit = normalize_text(meta.get("factuality_task_form")).lower()
    if explicit in {"single_answer", "pairwise"}:
        return explicit
    if sample.get("human_label") in PAIRWISE_FACTUALITY_LABELS:
        return "pairwise"
    if normalize_text(sample.get("answer_b")):
        return "pairwise"
    return "single_answer"


def factuality_decision_signal(sample: Dict[str, Any]) -> str:
    if sample.get("task_type") != "factuality_rag":
        return "non_factuality"
    if sample.get("human_label") in PAIRWISE_FACTUALITY_LABELS or _task_form(sample) == "pairwise":
        return "pairwise_factuality"
    if sample.get("human_label") in SINGLE_FACTUALITY_LABELS:
        return "single_answer_factuality"
    return "non_factuality"


def _answer_support(answer: Any, context: Any, reference: Any) -> Dict[str, float]:
    context_support = _support_score(answer, context)
    reference_support = _support_score(answer, reference)
    combined_evidence = f"{normalize_text(context)} {normalize_text(reference)}"
    combined_support = _support_score(answer, combined_evidence)
    claim_support = max(combined_support, 0.65 * context_support + 0.35 * reference_support)
    numeric_gap = _numeric_gap(answer, context, reference)
    date_gap = _date_gap(answer, context, reference)
    entity_gap = _entity_gap(answer, context, reference)
    entity_alias_gap = _entity_alias_gap(answer, context, reference)
    sentence_stats = _sentence_support_stats(answer, combined_evidence)
    anchored_sentence_stats = _anchored_sentence_gap_stats(answer, combined_evidence)
    negation_mismatch = _negation_mismatch(answer, combined_evidence)
    comparative_mismatch = _comparative_mismatch(answer, combined_evidence)
    local_hallucination_risk = max(
        sentence_stats["low_support_sentence_ratio"],
        entity_gap,
        entity_alias_gap,
        date_gap,
        numeric_gap,
        negation_mismatch,
        comparative_mismatch,
    )
    return {
        "context_support": clip01(context_support),
        "reference_support": clip01(reference_support),
        "claim_support_rate": clip01(claim_support),
        "numeric_evidence_gap": numeric_gap,
        "date_evidence_gap": date_gap,
        "entity_evidence_gap": entity_gap,
        "entity_alias_gap": entity_alias_gap,
        "negation_mismatch": negation_mismatch,
        "comparative_mismatch": comparative_mismatch,
        "sentence_max_support": sentence_stats["sentence_max_support"],
        "sentence_mean_support": sentence_stats["sentence_mean_support"],
        "low_support_sentence_ratio": sentence_stats["low_support_sentence_ratio"],
        "low_support_anchor_sentence_ratio": anchored_sentence_stats["low_support_anchor_sentence_ratio"],
        "max_low_support_anchor_gap": anchored_sentence_stats["max_low_support_anchor_gap"],
        "anchored_hallucination_severity": anchored_sentence_stats["anchored_hallucination_severity"],
        "local_hallucination_risk": clip01(local_hallucination_risk),
    }


def _pairwise_contradiction(label: Any, support_a: float, support_b: Optional[float]) -> Tuple[float, Optional[str]]:
    if support_b is None:
        return 0.0, None
    margin = support_a - support_b
    if label == "A>B" and margin < -0.05:
        return 1.0, "pairwise_support_contradiction"
    if label == "B>A" and margin > 0.05:
        return 1.0, "pairwise_support_contradiction"
    if label == "Tie" and abs(margin) > 0.35:
        return 0.5, "pairwise_support_tie_imbalance"
    return 0.0, None


def _risk_and_reasons(
    *,
    context: str,
    reference: str,
    form: str,
    label: Any,
    support_a: Dict[str, float],
    support_b: Optional[Dict[str, float]],
) -> Tuple[float, List[str], float]:
    reasons: List[str] = []
    risk_parts: List[float] = []

    if context and support_a["context_support"] < 0.35:
        reasons.append("low_context_support_a")
        risk_parts.append(0.70 * (1.0 - support_a["context_support"]))
    if reference and support_a["reference_support"] < 0.25:
        reasons.append("reference_support_gap_a")
        risk_parts.append(0.45 * (1.0 - support_a["reference_support"]))
    if support_a["numeric_evidence_gap"] > 0.0:
        reasons.append("numeric_evidence_gap_a")
        risk_parts.append(0.90 * support_a["numeric_evidence_gap"])
    if support_a["date_evidence_gap"] > 0.0:
        reasons.append("date_evidence_gap_a")
        risk_parts.append(0.85 * support_a["date_evidence_gap"])
    if support_a["entity_evidence_gap"] > 0.45:
        reasons.append("entity_evidence_gap_a")
        risk_parts.append(0.60 * support_a["entity_evidence_gap"])
    if support_a["entity_alias_gap"] > 0.45:
        reasons.append("entity_alias_gap_a")
        risk_parts.append(0.60 * support_a["entity_alias_gap"])
    if support_a["negation_mismatch"] > 0.0:
        reasons.append("negation_mismatch_a")
        risk_parts.append(0.85 * support_a["negation_mismatch"])
    if support_a["comparative_mismatch"] > 0.0:
        reasons.append("comparative_mismatch_a")
        risk_parts.append(0.65 * support_a["comparative_mismatch"])
    if support_a["low_support_sentence_ratio"] > 0.35:
        reasons.append("low_support_sentence_ratio_a")
        risk_parts.append(0.70 * support_a["low_support_sentence_ratio"])
    if support_a["low_support_anchor_sentence_ratio"] > 0.0:
        reasons.append("low_support_anchor_sentence_ratio_a")
    if support_a["max_low_support_anchor_gap"] > 0.0:
        reasons.append("max_low_support_anchor_gap_a")
    if support_a["anchored_hallucination_severity"] > 0.0:
        reasons.append("anchored_hallucination_severity_a")
    if support_a["local_hallucination_risk"] > 0.45:
        reasons.append("local_hallucination_risk_a")
        risk_parts.append(0.75 * support_a["local_hallucination_risk"])

    contradiction = 0.0
    if form == "pairwise" and support_b is not None:
        if context and support_b["context_support"] < 0.35:
            reasons.append("low_context_support_b")
            risk_parts.append(0.35 * (1.0 - support_b["context_support"]))
        if reference and support_b["reference_support"] < 0.25:
            reasons.append("reference_support_gap_b")
            risk_parts.append(0.25 * (1.0 - support_b["reference_support"]))
        if support_b["numeric_evidence_gap"] > 0.0:
            reasons.append("numeric_evidence_gap_b")
            risk_parts.append(0.45 * support_b["numeric_evidence_gap"])
        if support_b["date_evidence_gap"] > 0.0:
            reasons.append("date_evidence_gap_b")
            risk_parts.append(0.40 * support_b["date_evidence_gap"])
        if support_b["entity_alias_gap"] > 0.45:
            reasons.append("entity_alias_gap_b")
            risk_parts.append(0.30 * support_b["entity_alias_gap"])
        if support_b["negation_mismatch"] > 0.0:
            reasons.append("negation_mismatch_b")
            risk_parts.append(0.40 * support_b["negation_mismatch"])
        if support_b["comparative_mismatch"] > 0.0:
            reasons.append("comparative_mismatch_b")
            risk_parts.append(0.30 * support_b["comparative_mismatch"])
        if support_b["local_hallucination_risk"] > 0.45:
            reasons.append("local_hallucination_risk_b")
            risk_parts.append(0.45 * support_b["local_hallucination_risk"])
        if support_b["low_support_anchor_sentence_ratio"] > 0.0:
            reasons.append("low_support_anchor_sentence_ratio_b")
        if support_b["max_low_support_anchor_gap"] > 0.0:
            reasons.append("max_low_support_anchor_gap_b")
        contradiction, reason = _pairwise_contradiction(
            label,
            support_a["claim_support_rate"],
            support_b["claim_support_rate"],
        )
        if reason:
            reasons.append(reason)
            risk_parts.append(0.80 * contradiction)

    return clip01(max(risk_parts) if risk_parts else 0.0), sorted(set(reasons)), contradiction


def build_evidence_profile(sample: Dict[str, Any], *, review_threshold: float = 0.60) -> Dict[str, Any]:
    context = normalize_text(sample.get("context"))
    reference = normalize_text(sample.get("reference"))
    answer_a = normalize_text(sample.get("answer_a"))
    answer_b = normalize_text(sample.get("answer_b"))
    form = _task_form(sample)
    support_a = _answer_support(answer_a, context, reference)
    support_b = _answer_support(answer_b, context, reference) if form == "pairwise" and answer_b else None
    risk, reasons, contradiction = _risk_and_reasons(
        context=context,
        reference=reference,
        form=form,
        label=sample.get("human_label"),
        support_a=support_a,
        support_b=support_b,
    )
    support_delta = None if support_b is None else support_a["claim_support_rate"] - support_b["claim_support_rate"]

    return {
        "id": sample.get("id"),
        "dataset": sample.get("dataset"),
        "task_type": sample.get("task_type"),
        "split": sample.get("split"),
        "form": form,
        "label": sample.get("human_label"),
        "evidence": {
            "context_support_a": safe_round(support_a["context_support"]),
            "reference_support_a": safe_round(support_a["reference_support"]),
            "claim_support_rate_a": safe_round(support_a["claim_support_rate"]),
            "numeric_evidence_gap_a": safe_round(support_a["numeric_evidence_gap"]),
            "date_evidence_gap_a": safe_round(support_a["date_evidence_gap"]),
            "entity_evidence_gap_a": safe_round(support_a["entity_evidence_gap"]),
            "entity_alias_gap_a": safe_round(support_a["entity_alias_gap"]),
            "negation_mismatch_a": safe_round(support_a["negation_mismatch"]),
            "comparative_mismatch_a": safe_round(support_a["comparative_mismatch"]),
            "sentence_max_support_a": safe_round(support_a["sentence_max_support"]),
            "sentence_mean_support_a": safe_round(support_a["sentence_mean_support"]),
            "low_support_sentence_ratio_a": safe_round(support_a["low_support_sentence_ratio"]),
            "low_support_anchor_sentence_ratio_a": safe_round(support_a["low_support_anchor_sentence_ratio"]),
            "max_low_support_anchor_gap_a": safe_round(support_a["max_low_support_anchor_gap"]),
            "anchored_hallucination_severity_a": safe_round(support_a["anchored_hallucination_severity"]),
            "local_hallucination_risk_a": safe_round(support_a["local_hallucination_risk"]),
            "context_support_b": safe_round(support_b["context_support"]) if support_b else None,
            "reference_support_b": safe_round(support_b["reference_support"]) if support_b else None,
            "claim_support_rate_b": safe_round(support_b["claim_support_rate"]) if support_b else None,
            "numeric_evidence_gap_b": safe_round(support_b["numeric_evidence_gap"]) if support_b else None,
            "date_evidence_gap_b": safe_round(support_b["date_evidence_gap"]) if support_b else None,
            "entity_evidence_gap_b": safe_round(support_b["entity_evidence_gap"]) if support_b else None,
            "entity_alias_gap_b": safe_round(support_b["entity_alias_gap"]) if support_b else None,
            "negation_mismatch_b": safe_round(support_b["negation_mismatch"]) if support_b else None,
            "comparative_mismatch_b": safe_round(support_b["comparative_mismatch"]) if support_b else None,
            "sentence_max_support_b": safe_round(support_b["sentence_max_support"]) if support_b else None,
            "sentence_mean_support_b": safe_round(support_b["sentence_mean_support"]) if support_b else None,
            "low_support_sentence_ratio_b": safe_round(support_b["low_support_sentence_ratio"]) if support_b else None,
            "low_support_anchor_sentence_ratio_b": safe_round(support_b["low_support_anchor_sentence_ratio"]) if support_b else None,
            "max_low_support_anchor_gap_b": safe_round(support_b["max_low_support_anchor_gap"]) if support_b else None,
            "anchored_hallucination_severity_b": safe_round(support_b["anchored_hallucination_severity"]) if support_b else None,
            "local_hallucination_risk_b": safe_round(support_b["local_hallucination_risk"]) if support_b else None,
            "support_delta_a_minus_b": round(float(support_delta), 6) if support_delta is not None else None,
            "pairwise_support_contradiction": safe_round(contradiction),
            "evidence_risk": safe_round(risk),
            "review_required": risk >= review_threshold,
            "reasons": reasons,
        },
    }


def _feature_value(profile: Dict[str, Any], key: str) -> float:
    value = profile["evidence"].get(key)
    if value is None:
        return 0.0
    return float(value)


def evidence_feature_dict(sample: Dict[str, Any]) -> Dict[str, float]:
    profile = build_evidence_profile(sample)
    return {
        "evidence_context_support_a": _feature_value(profile, "context_support_a"),
        "evidence_reference_support_a": _feature_value(profile, "reference_support_a"),
        "evidence_claim_support_rate_a": _feature_value(profile, "claim_support_rate_a"),
        "evidence_numeric_gap_a": _feature_value(profile, "numeric_evidence_gap_a"),
        "evidence_date_gap_a": _feature_value(profile, "date_evidence_gap_a"),
        "evidence_entity_gap_a": _feature_value(profile, "entity_evidence_gap_a"),
        "evidence_entity_alias_gap_a": _feature_value(profile, "entity_alias_gap_a"),
        "evidence_negation_mismatch_a": _feature_value(profile, "negation_mismatch_a"),
        "evidence_comparative_mismatch_a": _feature_value(profile, "comparative_mismatch_a"),
        "evidence_sentence_max_support_a": _feature_value(profile, "sentence_max_support_a"),
        "evidence_sentence_mean_support_a": _feature_value(profile, "sentence_mean_support_a"),
        "evidence_low_support_sentence_ratio_a": _feature_value(profile, "low_support_sentence_ratio_a"),
        "evidence_low_support_anchor_sentence_ratio_a": _feature_value(
            profile,
            "low_support_anchor_sentence_ratio_a",
        ),
        "evidence_max_low_support_anchor_gap_a": _feature_value(profile, "max_low_support_anchor_gap_a"),
        "evidence_anchored_hallucination_severity_a": _feature_value(
            profile,
            "anchored_hallucination_severity_a",
        ),
        "evidence_local_hallucination_risk_a": _feature_value(profile, "local_hallucination_risk_a"),
        "evidence_context_support_b": _feature_value(profile, "context_support_b"),
        "evidence_reference_support_b": _feature_value(profile, "reference_support_b"),
        "evidence_claim_support_rate_b": _feature_value(profile, "claim_support_rate_b"),
        "evidence_numeric_gap_b": _feature_value(profile, "numeric_evidence_gap_b"),
        "evidence_date_gap_b": _feature_value(profile, "date_evidence_gap_b"),
        "evidence_entity_gap_b": _feature_value(profile, "entity_evidence_gap_b"),
        "evidence_entity_alias_gap_b": _feature_value(profile, "entity_alias_gap_b"),
        "evidence_negation_mismatch_b": _feature_value(profile, "negation_mismatch_b"),
        "evidence_comparative_mismatch_b": _feature_value(profile, "comparative_mismatch_b"),
        "evidence_sentence_max_support_b": _feature_value(profile, "sentence_max_support_b"),
        "evidence_sentence_mean_support_b": _feature_value(profile, "sentence_mean_support_b"),
        "evidence_low_support_sentence_ratio_b": _feature_value(profile, "low_support_sentence_ratio_b"),
        "evidence_low_support_anchor_sentence_ratio_b": _feature_value(
            profile,
            "low_support_anchor_sentence_ratio_b",
        ),
        "evidence_max_low_support_anchor_gap_b": _feature_value(profile, "max_low_support_anchor_gap_b"),
        "evidence_anchored_hallucination_severity_b": _feature_value(
            profile,
            "anchored_hallucination_severity_b",
        ),
        "evidence_local_hallucination_risk_b": _feature_value(profile, "local_hallucination_risk_b"),
        "evidence_support_delta_a_minus_b": _feature_value(profile, "support_delta_a_minus_b"),
        "evidence_pairwise_support_contradiction": _feature_value(profile, "pairwise_support_contradiction"),
        "evidence_risk": _feature_value(profile, "evidence_risk"),
        "evidence_review_required": 1.0 if profile["evidence"].get("review_required") else 0.0,
    }


def _accuracy_like(profiles: Iterable[Dict[str, Any]]) -> Optional[float]:
    labels = [profile.get("label") for profile in profiles]
    factuality = [label for label in labels if label in SINGLE_FACTUALITY_LABELS]
    if not factuality:
        return None
    supported = sum(1 for label in factuality if label == "supported")
    return round(supported / len(factuality), 4)


def _bucket_summary(profiles: List[Dict[str, Any]], key_fn) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        buckets[str(key_fn(profile) or "none")].append(profile)
    summary: Dict[str, Dict[str, Any]] = {}
    for key, rows in sorted(buckets.items()):
        summary[key] = {
            "count": len(rows),
            "supported_rate": _accuracy_like(rows),
            "review_rate": round(sum(1 for row in rows if row["evidence"]["review_required"]) / len(rows), 4),
            "avg_evidence_risk": round(
                sum(float(row["evidence"]["evidence_risk"]) for row in rows) / len(rows),
                4,
            ),
            "avg_claim_support_a": round(
                sum(float(row["evidence"]["claim_support_rate_a"]) for row in rows) / len(rows),
                4,
            ),
        }
    return summary


def summarize_evidence_profiles(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not profiles:
        return {
            "overall": {
                "profile_count": 0,
                "review_rate": 0.0,
                "avg_evidence_risk": 0.0,
                "avg_claim_support_a": 0.0,
            },
            "by_dataset": {},
            "by_form": {},
            "by_label": {},
            "reason_counts": {},
        }
    reason_counts = Counter(reason for profile in profiles for reason in profile["evidence"].get("reasons", []))
    return {
        "overall": {
            "profile_count": len(profiles),
            "review_rate": round(sum(1 for row in profiles if row["evidence"]["review_required"]) / len(profiles), 4),
            "avg_evidence_risk": round(
                sum(float(row["evidence"]["evidence_risk"]) for row in profiles) / len(profiles),
                4,
            ),
            "avg_claim_support_a": round(
                sum(float(row["evidence"]["claim_support_rate_a"]) for row in profiles) / len(profiles),
                4,
            ),
        },
        "by_dataset": _bucket_summary(profiles, lambda row: row.get("dataset")),
        "by_form": _bucket_summary(profiles, lambda row: row.get("form")),
        "by_label": _bucket_summary(profiles, lambda row: row.get("label")),
        "reason_counts": dict(reason_counts),
    }
