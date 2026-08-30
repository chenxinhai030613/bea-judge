"""
Standalone Chinese annotated dataset builder for BEA-Judge.

Outputs:
1) datasets/processed/chinese_professional_annotated_<N>.json
2) datasets/processed/chinese_professional_annotated_latest.json
3) datasets/splits_zh/{train,dev,test}.json
4) datasets/chinese_dataset_statistics.json
5) datasets/chinese_annotation_report.json
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataset_builder import (
    CORE_FIELDS,
    FACTUALITY_LABEL_VALUES,
    RNG as CORE_RNG,
    PAIRWISE_LABEL_VALUES,
    SEED,
    SPLIT_RATIOS,
    apply_task_field_contracts,
    assign_splits,
    balanced_select,
    base_sample,
    dataset_wrapper,
    invert_label,
    qc_and_finalize,
    split_group_key,
    stable_hash,
    stats_for,
    utc_now,
    write_json,
)


RNG = random.Random(SEED + 520)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "datasets"
PROCESSED_DIR = OUTPUT_DIR / "processed"
ZH_SPLIT_DIR = OUTPUT_DIR / "splits_zh"
DEFAULT_ZH_TARGET = 1000


DOMAINS = {
    "management": "管理",
    "finance": "财经",
    "policy": "政策",
    "research": "科研方法",
}


def pairwise_score(label: str) -> float:
    return PAIRWISE_LABEL_VALUES[label]


def factuality_score(label: str) -> float:
    mapped = {"A>B": "supported", "B>A": "unsupported", "Tie": "ambiguous"}[label]
    return FACTUALITY_LABEL_VALUES[mapped]


ORG_POOL = [
    "一家中型制造企业",
    "一家区域性商业银行",
    "一家省属国企",
    "一家互联网平台公司",
    "一家医药研发机构",
    "一家地方高校产研中心",
]

CHALLENGE_POOL = [
    "跨部门协同效率低",
    "项目交付延期率高",
    "预算执行偏差扩大",
    "政策落地反馈滞后",
    "研发里程碑反复变更",
    "数据口径不一致导致决策冲突",
]

OBJECTIVE_POOL = [
    "在六个月内将关键项目准时交付率提升到90%",
    "在两个季度内将预算偏差控制在5%以内",
    "在三个月内建立可复核的过程指标体系",
    "在一个评估周期内提升政策执行一致性",
    "在下个财年把风险事件发生率下降30%",
    "在半年内将跨团队返工率降低25%",
]

METRIC_POOL = [
    "延期工单占比",
    "一次通过率",
    "跨部门返工率",
    "预算执行偏差",
    "用户投诉率",
    "流程平均处理时长",
    "高风险事项暴露率",
    "周度里程碑偏差率",
]

FACT_CONTEXT_POOL = [
    {
        "domain": "finance",
        "context": "某上市公司2025年营业收入为128亿元，同比增长12%；经营活动现金流净额为18亿元，同比下降5%；研发投入占营收比例为6.5%。",
        "question": "根据材料，营业收入同比增速是多少？现金流变化提示了什么管理重点？",
        "correct": "营业收入同比增速为12%。现金流净额下降5%说明回款与营运资金管理需要重点跟踪，不能只看收入增长。",
        "wrong": "营业收入同比下降12%，现金流增长5%，说明可以减少回款管理力度。",
    },
    {
        "domain": "policy",
        "context": "某地数字治理试点要求：事项网上可办率不低于95%，群众满意度目标为90%，并在季度内完成两轮流程复盘。",
        "question": "试点的核心量化目标有哪些？",
        "correct": "量化目标包括网上可办率95%、满意度90%，以及季度内完成两轮流程复盘。",
        "wrong": "目标是网上可办率80%、满意度70%，流程复盘每年一次即可。",
    },
    {
        "domain": "management",
        "context": "某制造企业导入精益改进后，单位产线换线时间从52分钟下降到38分钟，月均报废率从3.2%降至2.5%。",
        "question": "改进措施的直接效果体现在哪些指标上？",
        "correct": "直接效果是换线时间缩短14分钟，月均报废率下降0.7个百分点，说明效率和质量均有改善。",
        "wrong": "换线时间延长到52分钟，报废率升至3.2%，说明改进无效。",
    },
    {
        "domain": "research",
        "context": "某实证研究样本量为1200，训练/验证/测试划分为70%/15%/15%。报告给出的AUC为0.81，ECE为0.07。",
        "question": "该研究在数据划分与校准表现上可得出什么结论？",
        "correct": "数据划分遵循70/15/15；AUC 0.81表明区分度较好，ECE 0.07说明概率校准误差相对较低。",
        "wrong": "研究没有划分验证集，AUC为0.18且ECE为0.70，模型无法使用。",
    },
]


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, PROCESSED_DIR, ZH_SPLIT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def cohen_kappa(labels1: List[str], labels2: List[str], categories: List[str]) -> float:
    if not labels1 or len(labels1) != len(labels2):
        return 0.0
    n = len(labels1)
    agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
    po = agree / n

    p1 = {c: labels1.count(c) / n for c in categories}
    p2 = {c: labels2.count(c) / n for c in categories}
    pe = sum(p1[c] * p2[c] for c in categories)
    if abs(1.0 - pe) < 1e-12:
        return 0.0
    return round((po - pe) / (1 - pe), 4)


def make_prompt(index: int, domain_key: str) -> Tuple[str, Dict[str, Any]]:
    org = ORG_POOL[index % len(ORG_POOL)]
    challenge = CHALLENGE_POOL[(index * 3) % len(CHALLENGE_POOL)]
    objective = OBJECTIVE_POOL[(index * 5) % len(OBJECTIVE_POOL)]
    metric = METRIC_POOL[(index * 7) % len(METRIC_POOL)]
    baseline = 8 + (index % 19)
    deadline_weeks = 8 + (index % 11)
    case_id = f"ZH-{index + 1:04d}"

    prompt = (
        f"{org}正在处理“{challenge}”问题，目标是{objective}。"
        f"当前{metric}约为{baseline}%，需要在{deadline_weeks}周内建立可复核追踪机制。"
        "请给出一份可执行方案，至少包含：问题诊断、行动路径、风险控制和评估指标。"
        f"（案例编号：{case_id}）"
    )
    return prompt, {
        "org": org,
        "challenge": challenge,
        "objective": objective,
        "metric": metric,
        "baseline": baseline,
        "deadline_weeks": deadline_weeks,
        "case_id": case_id,
        "domain": DOMAINS[domain_key],
    }


def answer_good(domain_label: str, info: Dict[str, Any]) -> str:
    return (
        f"建议采用“诊断-试点-推广-复盘”四步法推进{domain_label}改进："
        "第一，先统一口径并建立基线指标，明确责任人与时间表；"
        "第二，在1-2个高影响场景做小范围试点，按周复盘偏差；"
        "第三，把有效做法固化到流程和制度中，并同步培训；"
        "第四，设置预警阈值（进度、成本、质量）并建立纠偏机制。"
        f"针对当前场景，建议优先围绕“{info['challenge']}”建立里程碑看板，"
        f"把{info['metric']}从{info['baseline']}%压降，并在{info['deadline_weeks']}周内完成首轮闭环。"
    )


def answer_weak(domain_label: str, info: Dict[str, Any]) -> str:
    return (
        f"这个{domain_label}问题不复杂，先开会讨论一下就行。"
        f"可以先把{info['metric']}目标放一放，等数据自然好转再说。"
        "不需要细分责任和指标，也不用做阶段复盘，重点是保持团队心态稳定。"
    )


def answer_medium(domain_label: str, info: Dict[str, Any]) -> str:
    return (
        f"建议先制定{domain_label}改进计划，再逐步执行。"
        f"可以设定周会汇报、月度复盘，并关注成本、进度和{info['metric']}。"
        f"对于“{info['challenge']}”，先做流程梳理，再安排试点。"
    )


def make_annotation(label: str, sample_index: int) -> Dict[str, Any]:
    annotator1 = label
    annotator2 = label
    arbiter = None
    if label != "Tie" and sample_index % 7 == 0:
        annotator2 = invert_label(label) or label
        arbiter = label
    elif label == "Tie" and sample_index % 9 == 0:
        annotator2 = "A>B"
        arbiter = label
    agreement = int(annotator1 == annotator2)
    return {
        "annotator1": annotator1,
        "annotator2": annotator2,
        "arbiter": arbiter,
        "agreement": agreement,
    }


def build_open_qa_pool(target: int) -> List[Dict[str, Any]]:
    desired = max(int(target * 1.6), target + 120)
    samples: List[Dict[str, Any]] = []
    domain_keys = list(DOMAINS.keys())

    for idx in range(desired):
        domain_key = domain_keys[idx % len(domain_keys)]
        domain_label = DOMAINS[domain_key]
        prompt, info = make_prompt(idx, domain_key)

        strong = answer_good(domain_label, info)
        weak = answer_weak(domain_label, info)
        medium = answer_medium(domain_label, info)

        pattern = idx % 6
        if pattern in {0, 1, 2}:
            human_label = "A>B"
            answer_a, answer_b = strong, weak
        elif pattern in {3, 4}:
            human_label = "B>A"
            answer_a, answer_b = weak, strong
        else:
            human_label = "Tie"
            answer_a, answer_b = medium, medium + " 另外可补充风险台账与责任追踪。"

        ann = make_annotation(human_label, idx + 1)
        sample = base_sample(
            sample_id=f"zh_open_qa_{idx + 1:04d}",
            dataset="zh_professional_open_qa",
            task_type="open_qa",
            prompt=prompt,
            answer_a=answer_a,
            answer_b=answer_b,
            human_label=human_label,
            human_score={
                "score_format": "pairwise_votes_with_dimensions",
                "scoring_system": "pairwise_preference",
                "pairwise_preference": pairwise_score(human_label),
                "label": human_label,
                "annotator_votes": {
                    "annotator1": ann["annotator1"],
                    "annotator2": ann["annotator2"],
                    "arbiter": ann["arbiter"],
                },
                "dimension_scores_1_5": {
                    "relevance": 5 if human_label != "Tie" else 4,
                    "completeness": 5 if human_label == "A>B" else 2 if human_label == "B>A" else 4,
                    "factuality": 5 if human_label != "B>A" else 2,
                    "instruction_following": 5 if human_label != "B>A" else 2,
                    "clarity": 4 if human_label == "Tie" else 5 if human_label == "A>B" else 3,
                },
            },
            metadata={
                "source": "self_built_chinese_annotation",
                "domain": domain_label,
                "score_format": "pairwise_votes_with_dimensions",
                "scoring_system": "pairwise_preference",
                "annotation_protocol": "two_annotators_plus_arbiter_when_disagree",
                "annotator1_label": ann["annotator1"],
                "annotator2_label": ann["annotator2"],
                "arbiter_label": ann["arbiter"],
                "annotator_agreement": ann["agreement"],
            },
        )
        samples.append(sample)

    return samples


def format_bullets(text: str) -> str:
    pieces = [p.strip() for p in text.split("；") if p.strip()]
    if not pieces:
        return text
    return "格式化版本：\n" + "\n".join(f"- {p}" for p in pieces)


def build_pairwise_bias_pool(target: int) -> List[Dict[str, Any]]:
    desired_base = max(int(target * 0.45), 90)
    samples: List[Dict[str, Any]] = []
    domain_keys = list(DOMAINS.keys())
    perturbations = ("position", "length", "format", "rubric_sensitivity")

    for idx in range(desired_base):
        domain_key = domain_keys[idx % len(domain_keys)]
        domain_label = DOMAINS[domain_key]
        prompt, info = make_prompt(idx + 1000, domain_key)
        answer_a = answer_good(domain_label, info)
        answer_b = answer_weak(domain_label, info)
        label = "A>B" if idx % 2 == 0 else "B>A"
        if label == "B>A":
            answer_a, answer_b = answer_b, answer_a

        parent_id = f"zh_bias_parent_{idx + 1:04d}"
        for variant in perturbations:
            pa, pb, plabel, pprompt = answer_a, answer_b, label, prompt
            mapping = {"answer_a": "response_A", "answer_b": "response_B"}

            if variant == "position":
                pa, pb = pb, pa
                plabel = invert_label(plabel) or plabel
                mapping = {"answer_a": "original_response_B", "answer_b": "original_response_A"}
            elif variant == "length":
                filler = (
                    " 该补充段落仅增加字数，不引入新的证据。"
                    " 需要注意：篇幅变长不代表结论更可靠。"
                )
                if plabel == "A>B":
                    pb = pb + filler
                else:
                    pa = pa + filler
            elif variant == "format":
                if plabel == "A>B":
                    pb = format_bullets(pb.replace("。", "；"))
                else:
                    pa = format_bullets(pa.replace("。", "；"))
            else:
                pprompt = (
                    prompt
                    + "\n\n评审补充说明：优先判断推理可靠性与事实正确性，不因措辞华丽或篇幅更长而加分。"
                )

            ann = make_annotation(plabel, idx + 1)
            samples.append(
                base_sample(
                    sample_id=f"zh_pairwise_bias_{idx + 1:04d}_{variant}",
                    dataset="zh_professional_bias",
                    task_type="pairwise_bias",
                    prompt=pprompt,
                    answer_a=pa,
                    answer_b=pb,
                    human_label=plabel,
                    human_score={
                        "score_format": "pairwise_votes",
                        "scoring_system": "pairwise_preference",
                        "pairwise_preference": pairwise_score(plabel),
                        "label": plabel,
                        "annotator_votes": {
                            "annotator1": ann["annotator1"],
                            "annotator2": ann["annotator2"],
                            "arbiter": ann["arbiter"],
                        },
                    },
                    metadata={
                        "source": "self_built_chinese_annotation",
                        "domain": domain_label,
                        "score_format": "pairwise_votes",
                        "scoring_system": "pairwise_preference",
                        "parent_id": parent_id,
                        "bias_type": variant,
                        "perturbation_applied": variant,
                        "actual_mapping": mapping,
                        "annotation_protocol": "two_annotators_plus_arbiter_when_disagree",
                        "annotator1_label": ann["annotator1"],
                        "annotator2_label": ann["annotator2"],
                        "arbiter_label": ann["arbiter"],
                        "annotator_agreement": ann["agreement"],
                    },
                )
            )

    return samples


def build_factuality_pool(target: int) -> List[Dict[str, Any]]:
    desired = max(int(target * 1.4), target + 40)
    samples: List[Dict[str, Any]] = []

    for idx in range(desired):
        item = FACT_CONTEXT_POOL[idx % len(FACT_CONTEXT_POOL)]
        domain_label = DOMAINS[item["domain"]]
        case_id = f"ZHF-{idx + 1:04d}"
        prompt = f"{item['question']} 请基于给定材料回答，并说明依据。（案例编号：{case_id}）"
        context = f"{item['context']}（案例编号：{case_id}）"
        correct = f"{item['correct']}（依据案例：{case_id}）"
        wrong = f"{item['wrong']}（依据案例：{case_id}）"

        if idx % 3 == 0:
            label = "B>A"
            answer_a, answer_b = wrong, correct
        elif idx % 7 == 0:
            label = "Tie"
            answer_a = correct
            answer_b = correct + " 结论与原文一致，但仍建议补充数据更新时间。"
        else:
            label = "A>B"
            answer_a, answer_b = correct, wrong

        ann = make_annotation(label, idx + 1)
        samples.append(
            base_sample(
                sample_id=f"zh_factuality_{idx + 1:04d}",
                dataset="zh_professional_factuality",
                task_type="factuality_rag",
                prompt=prompt,
                context=context,
                answer_a=answer_a,
                answer_b=answer_b,
                reference=correct,
                human_label=label,
                human_score={
                    "score_format": "pairwise_factuality_votes",
                    "scoring_system": "pairwise_factuality",
                    "pairwise_preference": pairwise_score(label),
                    "factuality_label_score": factuality_score(label),
                    "label": label,
                    "faithfulness_a_1_5": 5 if label != "B>A" else 2,
                    "faithfulness_b_1_5": 5 if label == "B>A" else 2 if label == "A>B" else 4,
                    "annotator_votes": {
                        "annotator1": ann["annotator1"],
                        "annotator2": ann["annotator2"],
                        "arbiter": ann["arbiter"],
                    },
                },
                metadata={
                    "source": "self_built_chinese_annotation",
                    "domain": domain_label,
                    "score_format": "pairwise_factuality_votes",
                    "scoring_system": "pairwise_factuality",
                    "factuality_task_form": "pairwise",
                    "factuality_label": "supported"
                    if label == "A>B"
                    else "unsupported"
                    if label == "B>A"
                    else "ambiguous",
                    "context_relevance_label": 1,
                    "answer_relevance_label": 1 if label != "B>A" else 0,
                    "annotation_protocol": "two_annotators_plus_arbiter_when_disagree",
                    "annotator1_label": ann["annotator1"],
                    "annotator2_label": ann["annotator2"],
                    "arbiter_label": ann["arbiter"],
                    "annotator_agreement": ann["agreement"],
                },
            )
        )

    return samples


def build_chinese_samples(target: int = DEFAULT_ZH_TARGET) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target = int(target)
    if target < 120:
        raise RuntimeError("--target must be >= 120 to keep task/domain coverage.")

    RNG.seed(SEED + 520)
    CORE_RNG.seed(SEED)

    open_target = int(round(target * 0.40))
    bias_target = int(round(target * 0.40))
    factuality_target = target - open_target - bias_target

    open_pool = build_open_qa_pool(open_target)
    bias_pool = build_pairwise_bias_pool(bias_target)
    factuality_pool = build_factuality_pool(factuality_target)

    open_qc, open_reasons, open_removed = qc_and_finalize(open_pool, require_pair=True)
    bias_qc, bias_reasons, bias_removed = qc_and_finalize(bias_pool, require_pair=True)
    factuality_qc, factuality_reasons, factuality_removed = qc_and_finalize(factuality_pool, require_pair=True)

    if len(open_qc) < open_target:
        raise RuntimeError(f"open_qa pool too small after QC: need {open_target}, got {len(open_qc)}")
    if len(bias_qc) < bias_target:
        raise RuntimeError(f"pairwise_bias pool too small after QC: need {bias_target}, got {len(bias_qc)}")
    if len(factuality_qc) < factuality_target:
        raise RuntimeError(
            f"factuality_rag pool too small after QC: need {factuality_target}, got {len(factuality_qc)}"
        )

    open_final = balanced_select(open_qc, open_target, label_getter=lambda s: s.get("human_label") or "NA")
    bias_final = balanced_select(bias_qc, bias_target, label_getter=lambda s: s.get("human_label") or "NA")
    factuality_final = balanced_select(
        factuality_qc,
        factuality_target,
        label_getter=lambda s: s.get("metadata", {}).get("factuality_label") or s.get("human_label") or "NA",
    )

    if len(open_final) != open_target or len(bias_final) != bias_target or len(factuality_final) != factuality_target:
        raise RuntimeError("balanced selection did not reach requested target sizes.")

    all_samples = open_final + bias_final + factuality_final
    RNG.shuffle(all_samples)

    for idx, sample in enumerate(all_samples, 1):
        sample["id"] = f"zh_professional_{idx:04d}"
        sample["metadata"]["zh_group_key"] = split_group_key(sample)
        sample["metadata"]["sample_hash"] = stable_hash(
            sample.get("task_type", ""),
            sample.get("prompt", ""),
            sample.get("answer_a", ""),
            sample.get("answer_b", ""),
        )

    assign_splits(all_samples)
    apply_task_field_contracts(all_samples)

    build_meta = {
        "target": target,
        "selected_counts": {
            "open_qa": len(open_final),
            "pairwise_bias": len(bias_final),
            "factuality_rag": len(factuality_final),
        },
        "qc_removed": {
            "open_qa": dict(open_reasons),
            "pairwise_bias": dict(bias_reasons),
            "factuality_rag": dict(factuality_reasons),
        },
        "removed_log": {
            "open_qa": open_removed,
            "pairwise_bias": bias_removed,
            "factuality_rag": factuality_removed,
        },
    }
    return all_samples, build_meta


def write_zh_splits(samples: List[Dict[str, Any]]) -> None:
    for split in ("train", "dev", "test"):
        split_samples = [sample for sample in samples if sample.get("split") == split]
        payload = dataset_wrapper(
            f"BEA-Judge 中文标注集 {split} split",
            split_samples,
            ["self_built_chinese_annotation"],
        )
        write_json(ZH_SPLIT_DIR / f"{split}.json", payload)


def annotation_report(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    ann1: List[str] = []
    ann2: List[str] = []
    arbitration = 0

    for sample in samples:
        meta = sample.get("metadata", {})
        a1 = meta.get("annotator1_label")
        a2 = meta.get("annotator2_label")
        if a1 and a2:
            ann1.append(str(a1))
            ann2.append(str(a2))
        if meta.get("arbiter_label"):
            arbitration += 1

    categories = ["A>B", "B>A", "Tie"]
    disagreements = sum(1 for a, b in zip(ann1, ann2) if a != b)
    return {
        "created_at": utc_now(),
        "seed": SEED + 520,
        "sample_count": len(samples),
        "annotated_count": len(ann1),
        "disagreement_count": disagreements,
        "disagreement_rate": round(disagreements / len(ann1), 4) if ann1 else 0.0,
        "arbitration_count": arbitration,
        "cohen_kappa_annotator1_annotator2": cohen_kappa(ann1, ann2, categories),
        "label_distribution_final": dict(Counter(sample.get("human_label") for sample in samples)),
        "domain_distribution": dict(Counter(sample.get("metadata", {}).get("domain") for sample in samples)),
        "split_ratio_target": {
            "train": SPLIT_RATIOS[0],
            "dev": SPLIT_RATIOS[1],
            "test": SPLIT_RATIOS[2],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build standalone Chinese annotated dataset.")
    parser.add_argument(
        "--target",
        type=int,
        default=DEFAULT_ZH_TARGET,
        help=f"Total sample count (default: {DEFAULT_ZH_TARGET})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    all_samples, build_meta = build_chinese_samples(args.target)

    payload = dataset_wrapper(
        f"BEA-Judge 中文专业标注集 ({len(all_samples)})",
        all_samples,
        ["self_built_chinese_annotation"],
    )
    write_json(PROCESSED_DIR / f"chinese_professional_annotated_{len(all_samples)}.json", payload)
    write_json(PROCESSED_DIR / "chinese_professional_annotated_latest.json", payload)
    write_zh_splits(all_samples)

    stats = stats_for(all_samples, build_meta["qc_removed"])
    stats["required_fields"] = CORE_FIELDS
    stats["dataset_name"] = "BEA-Judge 中文专业标注集"
    stats["selected_counts"] = build_meta["selected_counts"]
    stats["missing_value_policy"] = "optional_empty_text_fields_use_null"
    write_json(OUTPUT_DIR / "chinese_dataset_statistics.json", stats)

    report = annotation_report(all_samples)
    report["qc_removed_log"] = build_meta["removed_log"]
    report["selected_counts"] = build_meta["selected_counts"]
    report["missing_value_policy"] = "optional_empty_text_fields_use_null"
    write_json(OUTPUT_DIR / "chinese_annotation_report.json", report)

    print(f"Built Chinese annotated dataset with {len(all_samples)} samples.")
    print(
        "open_qa={open_qa}, pairwise_bias={pairwise_bias}, factuality_rag={factuality_rag}".format(
            **build_meta["selected_counts"]
        )
    )
    print(f"kappa={report['cohen_kappa_annotator1_annotator2']}")


if __name__ == "__main__":
    main()
