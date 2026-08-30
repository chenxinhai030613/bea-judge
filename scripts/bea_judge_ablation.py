"""Run BEA-Judge module ablations with the existing lightweight trainer."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bea_judge_train import (  # noqa: E402
    FACTUALITY_LABELS,
    PAIRWISE_LABELS,
    JudgeOutputFeatures,
    base_pairwise_features,
    bias_features,
    bias_risk_features,
    disabled_tie_policy,
    evaluate_head_on_split,
    factuality_feature_dict,
    factuality_text_features,
    group_by_split,
    head_raw_probs_on_split,
    load_judge_output_features,
    make_calibrated_rows,
    metrics_for_calibrated_rows,
    one_hot_features,
    pairwise_feature_dict,
    read_dataset,
    select_samples,
    text_pairwise_features,
    train_one_head,
    validate_judge_output_coverage,
)
from evidence_features import evidence_feature_dict, factuality_decision_signal  # noqa: E402
from calibration_methods import SUPPORTED_METHODS, run_calibration_comparison  # noqa: E402


DATASETS = ROOT / "datasets"
REPORT_JSON = DATASETS / "model_outputs" / "latest_ablation_report.json"
REPORT_MD = DATASETS / "model_outputs" / "latest_ablation_report.md"
DEFAULT_MODEL_JSON = DATASETS / "model_outputs" / "bea_judge_20260521_110114" / "model.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def pairwise_without_bias(sample: Dict[str, Any], judge_outputs: JudgeOutputFeatures) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(text_pairwise_features(sample))
    features.update(base_pairwise_features(sample, judge_outputs))
    features.update(one_hot_features(sample))
    if factuality_decision_signal(sample) == "pairwise_factuality":
        features.update(evidence_feature_dict(sample))
    return features


def pairwise_without_evidence(sample: Dict[str, Any], judge_outputs: JudgeOutputFeatures) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(text_pairwise_features(sample))
    features.update(base_pairwise_features(sample, judge_outputs))
    features.update(bias_features(sample))
    features.update(bias_risk_features(sample))
    features.update(one_hot_features(sample))
    return features


def pairwise_without_base(sample: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(text_pairwise_features(sample))
    features.update(bias_features(sample))
    features.update(bias_risk_features(sample))
    features.update(one_hot_features(sample))
    if factuality_decision_signal(sample) == "pairwise_factuality":
        features.update(evidence_feature_dict(sample))
    return features


def pairwise_text_metadata_only(sample: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(text_pairwise_features(sample))
    features.update(one_hot_features(sample))
    return features


def pairwise_base_fusion_only(sample: Dict[str, Any], judge_outputs: JudgeOutputFeatures) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(text_pairwise_features(sample))
    features.update(base_pairwise_features(sample, judge_outputs))
    features.update(one_hot_features(sample))
    return features


def factuality_without_evidence(sample: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    features.update(factuality_text_features(sample))
    features.update(one_hot_features(sample))
    return features


def factuality_numeric_date_entity(sample: Dict[str, Any]) -> Dict[str, float]:
    features = factuality_without_evidence(sample)
    evidence = evidence_feature_dict(sample)
    for key, value in evidence.items():
        if any(token in key for token in ("numeric_gap", "date_gap", "entity_gap", "entity_alias_gap")):
            features[key] = value
    return features


def factuality_sentence_local_risk(sample: Dict[str, Any]) -> Dict[str, float]:
    features = factuality_numeric_date_entity(sample)
    evidence = evidence_feature_dict(sample)
    for key, value in evidence.items():
        if any(
            token in key
            for token in (
                "sentence_",
                "low_support_sentence_ratio",
                "low_support_anchor_sentence_ratio",
                "max_low_support_anchor_gap",
                "anchored_hallucination_severity",
                "local_hallucination_risk",
                "negation_mismatch",
                "comparative_mismatch",
                "claim_support_rate",
                "evidence_risk",
            )
        ):
            features[key] = value
    return features


def with_temperature_one(head: Dict[str, Any]) -> Dict[str, Any]:
    raw_head = copy.deepcopy(head)
    raw_head["calibration"]["temperature"] = 1.0
    return raw_head


def without_tie_policy(head: Dict[str, Any]) -> Dict[str, Any]:
    cloned = copy.deepcopy(head)
    cloned["tie_policy"] = disabled_tie_policy()
    return cloned


def without_review_threshold(head: Dict[str, Any]) -> Dict[str, Any]:
    cloned = copy.deepcopy(head)
    policy = copy.deepcopy(cloned.get("review_policy", {}))
    policy["method"] = "disabled_for_ablation"
    policy["threshold"] = 1.1
    policy["error_recall"] = 0.0
    policy["review_rate"] = 0.0
    policy["review_count"] = 0
    cloned["review_policy"] = policy
    return cloned


def run_variant(
    *,
    name: str,
    train: Sequence[Dict[str, Any]],
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    pairwise_feature_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    factuality_feature_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    factuality_active_labels: Sequence[str],
    no_calibration: bool = False,
) -> Dict[str, Any]:
    pair_train = select_samples(train, PAIRWISE_LABELS)
    pair_dev = select_samples(dev, PAIRWISE_LABELS)
    fact_train = select_samples(train, factuality_active_labels)
    fact_dev = select_samples(dev, factuality_active_labels)
    pair_head = train_one_head("pairwise", pair_train, pair_dev, PAIRWISE_LABELS, pairwise_feature_fn)
    fact_head = train_one_head("factuality", fact_train, fact_dev, factuality_active_labels, factuality_feature_fn)
    eval_pair_head = with_temperature_one(pair_head) if no_calibration else pair_head
    eval_fact_head = with_temperature_one(fact_head) if no_calibration else fact_head
    pair_test = evaluate_head_on_split(eval_pair_head, test, PAIRWISE_LABELS, pairwise_feature_fn, "test")
    fact_test = evaluate_head_on_split(eval_fact_head, test, factuality_active_labels, factuality_feature_fn, "test")
    pair_dev_rows = pair_head["calibrated_dev_rows"]
    fact_dev_rows = fact_head["calibrated_dev_rows"]
    return {
        "name": name,
        "pairwise": {
            "dev_metrics": pair_head["dev_metrics"] if no_calibration else pair_head["calibrated_dev_metrics"],
            "test_metrics": pair_test["metrics"],
            "dev_rows": pair_dev_rows,
            "test_rows": pair_test["rows"],
        },
        "factuality": {
            "dev_metrics": fact_head["dev_metrics"] if no_calibration else fact_head["calibrated_dev_metrics"],
            "test_metrics": fact_test["metrics"],
            "dev_rows": fact_dev_rows,
            "test_rows": fact_test["rows"],
        },
    }


def run_pairwise_variant(
    *,
    name: str,
    train: Sequence[Dict[str, Any]],
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    pairwise_feature_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    head_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    pair_train = select_samples(train, PAIRWISE_LABELS)
    pair_dev = select_samples(dev, PAIRWISE_LABELS)
    pair_head = train_one_head("pairwise", pair_train, pair_dev, PAIRWISE_LABELS, pairwise_feature_fn)
    eval_head = head_transform(pair_head) if head_transform is not None else pair_head
    pair_test = evaluate_head_on_split(eval_head, test, PAIRWISE_LABELS, pairwise_feature_fn, "test")
    pair_dev_eval = (
        None
        if head_transform is None
        else evaluate_head_on_split(eval_head, dev, PAIRWISE_LABELS, pairwise_feature_fn, "dev")
    )
    pair_dev_rows = eval_head["calibrated_dev_rows"] if pair_dev_eval is None else pair_dev_eval["rows"]
    pair_dev_metrics = eval_head["calibrated_dev_metrics"] if pair_dev_eval is None else pair_dev_eval["metrics"]
    return {
        "name": name,
        "pairwise": {
            "dev_metrics": pair_dev_metrics,
            "test_metrics": pair_test["metrics"],
            "dev_rows": pair_dev_rows,
            "test_rows": pair_test["rows"],
        },
    }


def run_pairwise_head_transform_variant(
    *,
    name: str,
    head: Dict[str, Any],
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    pairwise_feature_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    head_transform: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    eval_head = head_transform(head)
    pair_dev = evaluate_head_on_split(eval_head, dev, PAIRWISE_LABELS, pairwise_feature_fn, "dev")
    pair_test = evaluate_head_on_split(eval_head, test, PAIRWISE_LABELS, pairwise_feature_fn, "test")
    return {
        "name": name,
        "pairwise": {
            "dev_metrics": pair_dev["metrics"],
            "test_metrics": pair_test["metrics"],
            "dev_rows": pair_dev["rows"],
            "test_rows": pair_test["rows"],
        },
    }


def run_existing_heads_variant(
    *,
    name: str,
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    pairwise_head: Optional[Dict[str, Any]] = None,
    pairwise_feature_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    factuality_head: Optional[Dict[str, Any]] = None,
    factuality_feature_fn: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
    factuality_active_labels: Sequence[str] = FACTUALITY_LABELS,
    pairwise_head_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    factuality_head_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    control_type: Optional[str] = None,
) -> Dict[str, Any]:
    variant: Dict[str, Any] = {"name": name}
    if control_type:
        variant["control_type"] = control_type
    if pairwise_head is not None and pairwise_feature_fn is not None:
        eval_head = pairwise_head_transform(pairwise_head) if pairwise_head_transform else pairwise_head
        pair_dev = evaluate_head_on_split(eval_head, dev, PAIRWISE_LABELS, pairwise_feature_fn, "dev")
        pair_test = evaluate_head_on_split(eval_head, test, PAIRWISE_LABELS, pairwise_feature_fn, "test")
        variant["pairwise"] = {
            "dev_metrics": pair_dev["metrics"],
            "test_metrics": pair_test["metrics"],
            "dev_rows": pair_dev["rows"],
            "test_rows": pair_test["rows"],
        }
    if factuality_head is not None and factuality_feature_fn is not None:
        eval_head = factuality_head_transform(factuality_head) if factuality_head_transform else factuality_head
        fact_dev = evaluate_head_on_split(eval_head, dev, factuality_active_labels, factuality_feature_fn, "dev")
        fact_test = evaluate_head_on_split(eval_head, test, factuality_active_labels, factuality_feature_fn, "test")
        variant["factuality"] = {
            "dev_metrics": fact_dev["metrics"],
            "test_metrics": fact_test["metrics"],
            "dev_rows": fact_dev["rows"],
            "test_rows": fact_test["rows"],
        }
    return variant


def raw_base_rows(
    samples: Sequence[Dict[str, Any]],
    judge_outputs: JudgeOutputFeatures,
    split_name: str,
) -> List[Dict[str, Any]]:
    selected = select_samples(samples, PAIRWISE_LABELS)
    probs = []
    usable_samples = []
    pred_indices = []
    label_to_index = {label: i for i, label in enumerate(PAIRWISE_LABELS)}
    for sample in selected:
        row = judge_outputs.rows.get(str(sample.get("id")))
        if not row:
            continue
        pred_label = str(row.get("pred_label"))
        if pred_label not in label_to_index:
            continue
        usable_samples.append({**sample, "split": split_name})
        pred_indices.append(label_to_index[pred_label])
        probs.append([1.0 if label == pred_label else 0.0 for label in PAIRWISE_LABELS])
    if not usable_samples:
        return []
    rows = make_calibrated_rows(
        usable_samples,
        PAIRWISE_LABELS,
        np.array(probs, dtype=float),
        review_threshold=1.1,
        head_name="pairwise",
        pred_indices=np.array(pred_indices, dtype=int),
    )
    for row in rows:
        row["review_flag"] = False
        row["review_reason"] = "raw_base_no_review_policy"
    return rows


def run_raw_base_control(
    *,
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    judge_outputs: JudgeOutputFeatures,
) -> Dict[str, Any]:
    dev_rows = raw_base_rows(dev, judge_outputs, "dev")
    test_rows = raw_base_rows(test, judge_outputs, "test")
    return {
        "name": "Raw M-Prometheus-3B only",
        "control_type": "external_base_judge",
        "probability_policy": "one_hot_raw_pred_label",
        "pairwise": {
            "dev_metrics": metrics_for_calibrated_rows(dev_rows, PAIRWISE_LABELS),
            "test_metrics": metrics_for_calibrated_rows(test_rows, PAIRWISE_LABELS),
            "dev_rows": dev_rows,
            "test_rows": test_rows,
        },
    }


def bias_risk_only_review_rows(
    samples: Sequence[Dict[str, Any]],
    source_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sample_by_id = {str(sample.get("id")): sample for sample in samples}
    rows: List[Dict[str, Any]] = []
    for row in source_rows:
        sample = sample_by_id.get(str(row.get("id")))
        cloned = copy.deepcopy(row)
        if sample is None:
            cloned["review_flag"] = False
            cloned["review_reason"] = "auto_accept"
            rows.append(cloned)
            continue
        bias = bias_risk_features(sample)
        bias_indicators = bias_features(sample)
        reasons = []
        explicit_bias = any(
            float(bias_indicators.get(name, 0.0)) >= 1.0
            for name in ("bias_position", "bias_length", "bias_format", "bias_rubric")
        )
        if (
            bias.get("bias_review_required", 0.0) >= 1.0
            or float(bias.get("bias_overall_risk", 0.0)) >= 0.5
            or explicit_bias
        ):
            reasons.append("bias_risk_only_review")
        cloned["review_flag"] = bool(reasons)
        cloned["review_reason"] = "+".join(reasons) if reasons else "auto_accept"
        rows.append(cloned)
    return rows


def review_capture_rate(rows: Sequence[Dict[str, Any]]) -> Any:
    errors = [row for row in rows if row.get("human_label") != row.get("predicted_label")]
    if not errors:
        return ""
    return round(sum(1 for row in errors if row.get("review_flag")) / len(errors), 4)


def build_bias_utility(
    *,
    full_variant: Dict[str, Any],
    no_bias_variant: Dict[str, Any],
    train: Sequence[Dict[str, Any]],
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    all_samples = list(train) + list(dev) + list(test)
    no_bias_test_rows = no_bias_variant["pairwise"]["test_rows"]
    risk_only_rows = bias_risk_only_review_rows(all_samples, no_bias_test_rows)
    risk_only_metrics = metrics_for_calibrated_rows(risk_only_rows, PAIRWISE_LABELS)
    rows = []
    for name, variant_rows, metrics in (
        (
            "bias_as_decision_features",
            full_variant["pairwise"]["test_rows"],
            full_variant["pairwise"]["test_metrics"],
        ),
        (
            "no_bias_decision_features",
            no_bias_test_rows,
            no_bias_variant["pairwise"]["test_metrics"],
        ),
        (
            "bias_risk_only_review",
            risk_only_rows,
            risk_only_metrics,
        ),
    ):
        rows.append(
            {
                "setting": name,
                "head": "pairwise",
                "split": "test",
                "n": len(variant_rows),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "ece": metrics.get("ece"),
                "review_rate": round(sum(1 for row in variant_rows if row.get("review_flag")) / len(variant_rows), 4)
                if variant_rows
                else 0.0,
                "review_capture_rate": review_capture_rate(variant_rows),
            }
        )
    return rows


def run_factuality_feature_group_variant(
    *,
    name: str,
    train: Sequence[Dict[str, Any]],
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    factuality_feature_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    factuality_active_labels: Sequence[str],
    use_weighted_calibration: bool,
) -> Dict[str, Any]:
    fact_train = select_samples(train, factuality_active_labels)
    fact_dev = select_samples(dev, factuality_active_labels)
    weights = None if use_weighted_calibration else [{"class_weights": {}, "source_weights": {}}]
    fact_head = train_one_head(
        "factuality",
        fact_train,
        fact_dev,
        factuality_active_labels,
        factuality_feature_fn,
        weight_candidates_override=weights,
    )
    fact_test = evaluate_head_on_split(fact_head, test, factuality_active_labels, factuality_feature_fn, "test")
    return {
        "name": name,
        "weighted_calibration": use_weighted_calibration,
        "feature_count": len(fact_head["feature_names"]),
        "dev_metrics": fact_head["calibrated_dev_metrics"],
        "test_metrics": fact_test["metrics"],
    }


def factuality_feature_group_from_head(
    *,
    name: str,
    head: Dict[str, Any],
    test: Sequence[Dict[str, Any]],
    factuality_feature_fn: Callable[[Dict[str, Any]], Dict[str, float]],
    factuality_active_labels: Sequence[str],
    use_weighted_calibration: bool,
) -> Dict[str, Any]:
    fact_test = evaluate_head_on_split(head, test, factuality_active_labels, factuality_feature_fn, "test")
    return {
        "name": name,
        "weighted_calibration": use_weighted_calibration,
        "feature_count": len(head["feature_names"]),
        "dev_metrics": head["calibrated_dev_metrics"],
        "test_metrics": fact_test["metrics"],
    }


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# BEA-Judge Ablation Report",
        "",
        f"- Created at: {report['created_at']}",
        f"- Input dataset: `{report['input_dataset']}`",
        f"- Judge output: `{report['judge_output_path']}`",
        f"- Local prototype: `{report['local_prototype']}`",
        "",
        "## Variants",
        "",
    ]
    for variant in report["variants"]:
        pair = variant["pairwise"]["test_metrics"]
        fact = variant.get("factuality", {}).get("test_metrics", {})
        lines.append(
            f"- {variant['name']}: pairwise_acc={pair.get('accuracy')}, pairwise_f1={pair.get('macro_f1')}, "
            f"factuality_acc={fact.get('accuracy')}, factuality_f1={fact.get('macro_f1')}, "
            f"factuality_ece={fact.get('ece')}"
        )
    if report.get("control_baselines"):
        lines.extend(["", "## Control Baselines", ""])
        for variant in report["control_baselines"]:
            pair = variant.get("pairwise", {}).get("test_metrics", {})
            lines.append(
                f"- {variant['name']}: pairwise_acc={pair.get('accuracy')}, "
                f"pairwise_f1={pair.get('macro_f1')}, tie_recall={pair.get('tie_recall')}"
            )
    if report.get("feature_group_ablations"):
        lines.extend(["", "## Factuality Feature-Group Ablations", ""])
        for variant in report["feature_group_ablations"]:
            metrics = variant["test_metrics"]
            lines.append(
                f"- {variant['name']}: factuality_acc={metrics.get('accuracy')}, "
                f"factuality_f1={metrics.get('macro_f1')}, factuality_ece={metrics.get('ece')}"
            )
    return "\n".join(lines) + "\n"


def write_report(report: Dict[str, Any], *, report_json: Path = REPORT_JSON, report_md: Optional[Path] = None) -> None:
    report_md = report_md or report_json.with_suffix(".md")
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(build_markdown(report), encoding="utf-8")


def dedupe_variants(variants: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for variant in variants:
        name = str(variant.get("name"))
        if name in seen:
            continue
        seen.add(name)
        out.append(variant)
    return out


def limit_split_samples(
    train: Sequence[Dict[str, Any]],
    dev: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    sample_limit: Optional[int],
) -> Tuple[Sequence[Dict[str, Any]], Sequence[Dict[str, Any]], Sequence[Dict[str, Any]]]:
    if sample_limit is None or sample_limit <= 0:
        return train, dev, test
    limit = int(sample_limit)
    return train[:limit], dev[:limit], test[:limit]


def extend_existing_report(
    *,
    existing_report_path: Path,
    model_json_path: Path,
    input_path: Path,
    judge_output_path: Path,
    allow_heuristic_base: bool,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    report = json.loads(existing_report_path.read_text(encoding="utf-8"))
    samples = read_dataset(input_path)
    judge_outputs = load_judge_output_features(judge_output_path, allow_heuristic=allow_heuristic_base)
    splits = group_by_split(samples)
    train = splits["train"]
    dev = splits["dev"]
    test = splits["test"]
    train, dev, test = limit_split_samples(train, dev, test, sample_limit)
    pair_head = json.loads(model_json_path.read_text(encoding="utf-8"))["pairwise"]
    full_pairwise = lambda sample: pairwise_feature_dict(sample, judge_outputs)
    base_fusion_pairwise = lambda sample: pairwise_base_fusion_only(sample, judge_outputs)

    variants = [
        variant
        for variant in report.get("variants", [])
        if str(variant.get("name"))
        not in {"w/o Base Judge Scores", "w/o Tie Policy", "w/o Review Threshold"}
    ]
    variants.extend(
        [
            run_existing_heads_variant(
                name="w/o Base Judge Scores",
                dev=dev,
                test=test,
                pairwise_head=pair_head,
                pairwise_feature_fn=pairwise_without_base,
                control_type="inference_feature_ablation",
            ),
            run_pairwise_head_transform_variant(
                name="w/o Tie Policy",
                head=pair_head,
                dev=dev,
                test=test,
                pairwise_feature_fn=full_pairwise,
                head_transform=without_tie_policy,
            ),
            run_pairwise_head_transform_variant(
                name="w/o Review Threshold",
                head=pair_head,
                dev=dev,
                test=test,
                pairwise_feature_fn=full_pairwise,
                head_transform=without_review_threshold,
            ),
        ]
    )
    control_baselines = [
        run_raw_base_control(dev=dev, test=test, judge_outputs=judge_outputs),
        run_existing_heads_variant(
            name="Text/metadata-only",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=pairwise_text_metadata_only,
            control_type="inference_feature_ablation",
        ),
        run_existing_heads_variant(
            name="Base + fusion calibration only",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=base_fusion_pairwise,
            control_type="inference_feature_ablation",
        ),
    ]
    raw_dev_probs, y_dev_pair = head_raw_probs_on_split(pair_head, dev, PAIRWISE_LABELS, full_pairwise)
    raw_test_probs, y_test_pair = head_raw_probs_on_split(pair_head, test, PAIRWISE_LABELS, full_pairwise)
    calibration_methods = (
        run_calibration_comparison(
            p_dev=raw_dev_probs,
            y_dev=y_dev_pair,
            p_test=raw_test_probs,
            y_test=y_test_pair,
            methods=SUPPORTED_METHODS,
            out_dir=DATASETS / "model_outputs" / "latest_calibration_comparison",
            head="pairwise",
        )
        if raw_dev_probs.shape[0] and raw_test_probs.shape[0]
        else {}
    )
    variants = dedupe_variants(variants)
    full_variant = next(row for row in variants if row.get("name") == "Full BEA-Judge")
    no_bias_variant = next(row for row in variants if row.get("name") == "w/o Bias Module")
    report.update(
        {
            "created_at": utc_now(),
            "input_dataset": path_relative_to_root(input_path),
            "judge_output_path": path_relative_to_root(judge_output_path),
            "model_json_path": path_relative_to_root(model_json_path),
            "extension_mode": "reuse_existing_ablation_report_with_frozen_model",
            "variants": variants,
            "control_baselines": control_baselines,
            "calibration_methods": calibration_methods,
            "bias_utility": build_bias_utility(
                full_variant=full_variant,
                no_bias_variant=no_bias_variant,
                train=train,
                dev=dev,
                test=test,
            ),
        }
    )
    return report


def build_reuse_model_report(
    *,
    model_json_path: Path,
    input_path: Path,
    judge_output_path: Path,
    allow_heuristic_base: bool,
    sample_limit: Optional[int] = None,
) -> Dict[str, Any]:
    samples = read_dataset(input_path)
    judge_outputs = load_judge_output_features(judge_output_path, allow_heuristic=allow_heuristic_base)
    splits = group_by_split(samples)
    train = splits["train"]
    dev = splits["dev"]
    test = splits["test"]
    train, dev, test = limit_split_samples(train, dev, test, sample_limit)
    validate_judge_output_coverage(
        select_samples(train, PAIRWISE_LABELS)
        + select_samples(dev, PAIRWISE_LABELS)
        + select_samples(test, PAIRWISE_LABELS),
        judge_outputs,
    )
    model = json.loads(model_json_path.read_text(encoding="utf-8"))
    pair_head = model["pairwise"]
    fact_head = model.get("factuality")
    factuality_active_labels = list(fact_head.get("labels", FACTUALITY_LABELS)) if fact_head else []
    full_pairwise = lambda sample: pairwise_feature_dict(sample, judge_outputs)
    no_bias_pairwise = lambda sample: pairwise_without_bias(sample, judge_outputs)
    no_evidence_pairwise = lambda sample: pairwise_without_evidence(sample, judge_outputs)
    base_fusion_pairwise = lambda sample: pairwise_base_fusion_only(sample, judge_outputs)

    variants = [
        run_existing_heads_variant(
            name="Full BEA-Judge",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=full_pairwise,
            factuality_head=fact_head,
            factuality_feature_fn=factuality_feature_dict if fact_head else None,
            factuality_active_labels=factuality_active_labels,
            control_type="reuse_trained_head",
        ),
        run_existing_heads_variant(
            name="w/o Bias Module",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=no_bias_pairwise,
            factuality_head=fact_head,
            factuality_feature_fn=factuality_feature_dict if fact_head else None,
            factuality_active_labels=factuality_active_labels,
            control_type="inference_feature_ablation",
        ),
        run_existing_heads_variant(
            name="w/o Evidence Module",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=no_evidence_pairwise,
            factuality_head=fact_head,
            factuality_feature_fn=factuality_without_evidence if fact_head else None,
            factuality_active_labels=factuality_active_labels,
            control_type="inference_feature_ablation",
        ),
        run_existing_heads_variant(
            name="w/o Calibration",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=full_pairwise,
            factuality_head=fact_head,
            factuality_feature_fn=factuality_feature_dict if fact_head else None,
            factuality_active_labels=factuality_active_labels,
            pairwise_head_transform=with_temperature_one,
            factuality_head_transform=with_temperature_one,
            control_type="head_policy_ablation",
        ),
        run_existing_heads_variant(
            name="w/o Base Judge Scores",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=pairwise_without_base,
            control_type="inference_feature_ablation",
        ),
        run_pairwise_head_transform_variant(
            name="w/o Tie Policy",
            head=pair_head,
            dev=dev,
            test=test,
            pairwise_feature_fn=full_pairwise,
            head_transform=without_tie_policy,
        ),
        run_pairwise_head_transform_variant(
            name="w/o Review Threshold",
            head=pair_head,
            dev=dev,
            test=test,
            pairwise_feature_fn=full_pairwise,
            head_transform=without_review_threshold,
        ),
    ]
    control_baselines = [
        run_raw_base_control(dev=dev, test=test, judge_outputs=judge_outputs),
        run_existing_heads_variant(
            name="Text/metadata-only",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=pairwise_text_metadata_only,
            control_type="inference_feature_ablation",
        ),
        run_existing_heads_variant(
            name="Base + fusion calibration only",
            dev=dev,
            test=test,
            pairwise_head=pair_head,
            pairwise_feature_fn=base_fusion_pairwise,
            control_type="inference_feature_ablation",
        ),
    ]
    feature_group_ablations: List[Dict[str, Any]] = []
    if fact_head:
        feature_group_ablations = [
            factuality_feature_group_from_head(
                name="overlap-only",
                head=fact_head,
                test=test,
                factuality_feature_fn=factuality_without_evidence,
                factuality_active_labels=factuality_active_labels,
                use_weighted_calibration=False,
            ),
            factuality_feature_group_from_head(
                name="+numeric/date/entity",
                head=fact_head,
                test=test,
                factuality_feature_fn=factuality_numeric_date_entity,
                factuality_active_labels=factuality_active_labels,
                use_weighted_calibration=False,
            ),
            factuality_feature_group_from_head(
                name="+sentence/local-risk",
                head=fact_head,
                test=test,
                factuality_feature_fn=factuality_sentence_local_risk,
                factuality_active_labels=factuality_active_labels,
                use_weighted_calibration=False,
            ),
            factuality_feature_group_from_head(
                name="+weighted calibration",
                head=fact_head,
                test=test,
                factuality_feature_fn=factuality_feature_dict,
                factuality_active_labels=factuality_active_labels,
                use_weighted_calibration=True,
            ),
        ]
    full_variant = next(row for row in variants if row.get("name") == "Full BEA-Judge")
    no_bias_variant = next(row for row in variants if row.get("name") == "w/o Bias Module")
    return {
        "created_at": utc_now(),
        "input_dataset": path_relative_to_root(input_path),
        "judge_output_path": path_relative_to_root(judge_output_path),
        "model_json_path": path_relative_to_root(model_json_path),
        "local_prototype": bool(allow_heuristic_base),
        "sample_limit": sample_limit,
        "replay_mode": "reuse_existing_model_inference_feature_ablation",
        "variants": variants,
        "control_baselines": control_baselines,
        "calibration_methods": {},
        "bias_utility": build_bias_utility(
            full_variant=full_variant,
            no_bias_variant=no_bias_variant,
            train=train,
            dev=dev,
            test=test,
        ),
        "feature_group_ablations": feature_group_ablations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BEA-Judge module ablations.")
    parser.add_argument("--input", type=Path, default=DATASETS / "processed" / "bea_judge_cleaned_3400.json")
    parser.add_argument("--judge-output", type=Path, required=True)
    parser.add_argument("--allow-heuristic-base", action="store_true")
    parser.add_argument("--extend-existing-report", type=Path, default=None)
    parser.add_argument("--model-json", type=Path, default=DEFAULT_MODEL_JSON)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=None)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional per-split sample cap for smoke tests. Leave unset for formal reports.",
    )
    parser.add_argument(
        "--reuse-model-only",
        action="store_true",
        help="Reuse an existing model.json for inference-level ablations without retraining lightweight heads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_model_only:
        report = build_reuse_model_report(
            model_json_path=args.model_json,
            input_path=args.input,
            judge_output_path=args.judge_output,
            allow_heuristic_base=args.allow_heuristic_base,
            sample_limit=args.sample_limit,
        )
        write_report(report, report_json=args.report_json, report_md=args.report_md)
        print(
            json.dumps(
                {
                    "variants": [variant["name"] for variant in report["variants"]],
                    "control_baselines": [variant["name"] for variant in report.get("control_baselines", [])],
                    "report": str(args.report_json.resolve()),
                    "replay_mode": report.get("replay_mode"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.extend_existing_report:
        report = extend_existing_report(
            existing_report_path=args.extend_existing_report,
            model_json_path=args.model_json,
            input_path=args.input,
            judge_output_path=args.judge_output,
            allow_heuristic_base=args.allow_heuristic_base,
            sample_limit=args.sample_limit,
        )
        write_report(report, report_json=args.report_json, report_md=args.report_md)
        print(
            json.dumps(
                {
                    "variants": [variant["name"] for variant in report["variants"]],
                    "control_baselines": [variant["name"] for variant in report.get("control_baselines", [])],
                    "report": str(args.report_json.resolve()),
                    "extension_mode": report.get("extension_mode"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    samples = read_dataset(args.input)
    judge_outputs = load_judge_output_features(args.judge_output, allow_heuristic=args.allow_heuristic_base)
    splits = group_by_split(samples)
    train = splits["train"]
    dev = splits["dev"]
    test = splits["test"]
    train, dev, test = limit_split_samples(train, dev, test, args.sample_limit)
    validate_judge_output_coverage(
        select_samples(train, PAIRWISE_LABELS)
        + select_samples(dev, PAIRWISE_LABELS)
        + select_samples(test, PAIRWISE_LABELS),
        judge_outputs,
    )
    factuality_active_labels = [
        label
        for label in FACTUALITY_LABELS
        if any(sample.get("human_label") == label for sample in select_samples(train + dev, FACTUALITY_LABELS))
    ]
    full_pairwise = lambda sample: pairwise_feature_dict(sample, judge_outputs)
    no_bias_pairwise = lambda sample: pairwise_without_bias(sample, judge_outputs)
    no_evidence_pairwise = lambda sample: pairwise_without_evidence(sample, judge_outputs)
    base_fusion_pairwise = lambda sample: pairwise_base_fusion_only(sample, judge_outputs)
    pair_train = select_samples(train, PAIRWISE_LABELS)
    pair_dev = select_samples(dev, PAIRWISE_LABELS)
    fact_train = select_samples(train, factuality_active_labels)
    fact_dev = select_samples(dev, factuality_active_labels)
    full_pair_head = train_one_head("pairwise", pair_train, pair_dev, PAIRWISE_LABELS, full_pairwise)
    full_fact_head = train_one_head("factuality", fact_train, fact_dev, factuality_active_labels, factuality_feature_dict)
    no_bias_pair_head = train_one_head("pairwise", pair_train, pair_dev, PAIRWISE_LABELS, no_bias_pairwise)
    no_evidence_pair_head = train_one_head("pairwise", pair_train, pair_dev, PAIRWISE_LABELS, no_evidence_pairwise)
    no_evidence_fact_head = train_one_head(
        "factuality",
        fact_train,
        fact_dev,
        factuality_active_labels,
        factuality_without_evidence,
    )
    variants = [
        run_existing_heads_variant(
            name="Full BEA-Judge",
            dev=dev,
            test=test,
            pairwise_head=full_pair_head,
            pairwise_feature_fn=full_pairwise,
            factuality_head=full_fact_head,
            factuality_feature_fn=factuality_feature_dict,
            factuality_active_labels=factuality_active_labels,
        ),
        run_existing_heads_variant(
            name="w/o Bias Module",
            dev=dev,
            test=test,
            pairwise_head=no_bias_pair_head,
            pairwise_feature_fn=no_bias_pairwise,
            factuality_head=full_fact_head,
            factuality_feature_fn=factuality_feature_dict,
            factuality_active_labels=factuality_active_labels,
        ),
        run_existing_heads_variant(
            name="w/o Evidence Module",
            dev=dev,
            test=test,
            pairwise_head=no_evidence_pair_head,
            pairwise_feature_fn=no_evidence_pairwise,
            factuality_head=no_evidence_fact_head,
            factuality_feature_fn=factuality_without_evidence,
            factuality_active_labels=factuality_active_labels,
        ),
        run_existing_heads_variant(
            name="w/o Calibration",
            dev=dev,
            test=test,
            pairwise_head=full_pair_head,
            pairwise_feature_fn=full_pairwise,
            factuality_head=full_fact_head,
            factuality_feature_fn=factuality_feature_dict,
            factuality_active_labels=factuality_active_labels,
            pairwise_head_transform=with_temperature_one,
            factuality_head_transform=with_temperature_one,
        ),
    ]
    variants.extend(
        [
            run_existing_heads_variant(
                name="w/o Base Judge Scores",
                dev=dev,
                test=test,
                pairwise_head=full_pair_head,
                pairwise_feature_fn=pairwise_without_base,
                control_type="inference_feature_ablation",
            ),
            run_pairwise_head_transform_variant(
                name="w/o Tie Policy",
                head=full_pair_head,
                dev=dev,
                test=test,
                pairwise_feature_fn=full_pairwise,
                head_transform=without_tie_policy,
            ),
            run_pairwise_head_transform_variant(
                name="w/o Review Threshold",
                head=full_pair_head,
                dev=dev,
                test=test,
                pairwise_feature_fn=full_pairwise,
                head_transform=without_review_threshold,
            ),
        ]
    )
    control_baselines = [
        run_raw_base_control(dev=dev, test=test, judge_outputs=judge_outputs),
        run_existing_heads_variant(
            name="Text/metadata-only",
            dev=dev,
            test=test,
            pairwise_head=full_pair_head,
            pairwise_feature_fn=pairwise_text_metadata_only,
            control_type="inference_feature_ablation",
        ),
        run_existing_heads_variant(
            name="Base + fusion calibration only",
            dev=dev,
            test=test,
            pairwise_head=full_pair_head,
            pairwise_feature_fn=base_fusion_pairwise,
            control_type="inference_feature_ablation",
        ),
    ]
    raw_dev_probs, y_dev_pair = head_raw_probs_on_split(full_pair_head, dev, PAIRWISE_LABELS, full_pairwise)
    raw_test_probs, y_test_pair = head_raw_probs_on_split(full_pair_head, test, PAIRWISE_LABELS, full_pairwise)
    calibration_methods = (
        run_calibration_comparison(
            p_dev=raw_dev_probs,
            y_dev=y_dev_pair,
            p_test=raw_test_probs,
            y_test=y_test_pair,
            methods=SUPPORTED_METHODS,
            out_dir=DATASETS / "model_outputs" / "latest_calibration_comparison",
            head="pairwise",
        )
        if raw_dev_probs.shape[0] and raw_test_probs.shape[0]
        else {}
    )
    feature_group_ablations = [
        factuality_feature_group_from_head(
            name="overlap-only",
            head=no_evidence_fact_head,
            test=test,
            factuality_feature_fn=factuality_without_evidence,
            factuality_active_labels=factuality_active_labels,
            use_weighted_calibration=False,
        ),
        run_factuality_feature_group_variant(
            name="+numeric/date/entity",
            train=train,
            dev=dev,
            test=test,
            factuality_feature_fn=factuality_numeric_date_entity,
            factuality_active_labels=factuality_active_labels,
            use_weighted_calibration=False,
        ),
        run_factuality_feature_group_variant(
            name="+sentence/local-risk",
            train=train,
            dev=dev,
            test=test,
            factuality_feature_fn=factuality_sentence_local_risk,
            factuality_active_labels=factuality_active_labels,
            use_weighted_calibration=False,
        ),
        factuality_feature_group_from_head(
            name="+weighted calibration",
            head=full_fact_head,
            test=test,
            factuality_feature_fn=factuality_feature_dict,
            factuality_active_labels=factuality_active_labels,
            use_weighted_calibration=True,
        ),
    ]
    report = {
        "created_at": utc_now(),
        "input_dataset": path_relative_to_root(args.input),
        "judge_output_path": path_relative_to_root(args.judge_output),
        "local_prototype": bool(args.allow_heuristic_base),
        "variants": variants,
        "control_baselines": control_baselines,
        "calibration_methods": calibration_methods,
        "bias_utility": build_bias_utility(
            full_variant=variants[0],
            no_bias_variant=variants[1],
            train=train,
            dev=dev,
            test=test,
        ),
        "feature_group_ablations": feature_group_ablations,
    }
    write_report(report, report_json=args.report_json, report_md=args.report_md)
    print(
        json.dumps(
            {
                "variants": [variant["name"] for variant in variants],
                "report": str(args.report_json.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
