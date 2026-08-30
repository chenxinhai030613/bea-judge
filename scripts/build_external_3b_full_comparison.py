"""Build the integrated external lightweight baseline comparison table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
METRICS = ("accuracy", "macro_f1", "ece", "tie_recall")
DEFAULT_OUTPUT_DIR = "datasets/model_outputs/external_3b_baseline_comparison"


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_block(mean: Optional[float], std: Optional[float] = None, n: Optional[int] = None) -> Dict[str, Any]:
    return {
        "mean": None if mean is None else round(float(mean), 6),
        "std": None if std is None else round(float(std), 6),
        "n": n,
    }


def epoch_setting(summary: Mapping[str, Any], setting: str) -> Mapping[str, Any]:
    for item in summary.get("results", []):
        if item.get("setting") == setting:
            return item
    raise KeyError(f"setting not found in epoch summary: {setting}")


def epoch_system_metrics(summary: Mapping[str, Any], setting: str, system: str) -> Dict[str, Dict[str, Any]]:
    payload = epoch_setting(summary, setting).get("by_system", {})
    if system not in payload:
        raise KeyError(f"system not found for {setting}: {system}")
    return {
        metric: metric_block(
            payload[system].get(metric, {}).get("mean"),
            payload[system].get(metric, {}).get("std"),
            payload[system].get(metric, {}).get("n"),
        )
        for metric in METRICS
    }


def external_system_metrics(report: Mapping[str, Any], model_name: str) -> Dict[str, Dict[str, Any]]:
    payload = report.get("baselines", {})
    if model_name not in payload:
        raise KeyError(f"external baseline not found: {model_name}")
    metrics = payload[model_name].get("test_metrics", {})
    return {metric: metric_block(metrics.get(metric), None, metrics.get("n")) for metric in METRICS}


def rescue_metrics(audit: Mapping[str, Any], setting: str) -> Dict[str, Dict[str, Any]]:
    for item in audit.get("results", []):
        if item.get("setting") == setting:
            test = item.get("mean_std", {}).get("test", {})
            return {
                metric: metric_block(test.get(metric, {}).get("mean"), test.get(metric, {}).get("std"), 3)
                for metric in METRICS
            }
    raise KeyError(f"setting not found in tie rescue audit: {setting}")


def metric_mean(metrics: Mapping[str, Mapping[str, Any]], metric: str) -> Optional[float]:
    value = metrics.get(metric, {}).get("mean")
    return None if value is None else float(value)


def metric_cell(block: Mapping[str, Any]) -> str:
    if block.get("mean") is None:
        return ""
    mean = float(block["mean"])
    if block.get("std") is None:
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {float(block.get('std') or 0.0):.4f}"


def make_row(
    *,
    system: str,
    role: str,
    run_type: str,
    training: str,
    four_module: str,
    calibration: str,
    metrics: Mapping[str, Mapping[str, Any]],
    parse_failure_rate: Optional[float] = None,
    n: Optional[int] = None,
    source: str,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "system": system,
        "role": role,
        "run_type": run_type,
        "training": training,
        "four_module": four_module,
        "calibration": calibration,
        "metrics": {metric: dict(metrics.get(metric, {})) for metric in METRICS},
        "parse_failure_rate": parse_failure_rate,
        "n": n,
        "source": source,
        "notes": notes,
    }


def external_parse_failure(report: Mapping[str, Any], model_name: str) -> Optional[float]:
    metrics = report.get("baselines", {}).get(model_name, {}).get("test_metrics", {})
    value = metrics.get("parse_failure_rate")
    return None if value is None else float(value)


def external_n(report: Mapping[str, Any], model_name: str) -> Optional[int]:
    metrics = report.get("baselines", {}).get(model_name, {}).get("test_metrics", {})
    value = metrics.get("n")
    return None if value is None else int(value)


def build_summary(
    *,
    epoch_summary: Mapping[str, Any],
    external_report: Mapping[str, Any],
    tie_rescue_audit: Mapping[str, Any],
    setting: str = "epoch2_1024",
) -> Dict[str, Any]:
    grm_name = "Ray2333/GRM-Llama3.2-3B-rewardmodel-ft"
    qwen_name = "Qwen/Qwen2.5-3B-Instruct"
    glider_name = "PatronusAI/glider"
    rows = [
        make_row(
            system="Raw M-Prometheus-3B",
            role="internal frozen base",
            run_type="3-seed repeated baseline",
            training="frozen",
            four_module="no",
            calibration="none",
            metrics=epoch_system_metrics(epoch_summary, setting, "Raw M-Prometheus-3B"),
            n=3,
            source="qlora_epoch_ablation_3seed_1024_summary",
        ),
        make_row(
            system="Current BEA-Judge",
            role="internal four-module baseline",
            run_type="3-seed repeated baseline",
            training="frozen",
            four_module="yes",
            calibration="fusion calibration",
            metrics=epoch_system_metrics(epoch_summary, setting, "Current BEA-Judge"),
            n=3,
            source="qlora_epoch_ablation_3seed_1024_summary",
        ),
        make_row(
            system="QLoRA-M-Prometheus-3B",
            role="internal QLoRA base",
            run_type="3-seed mean+/-std",
            training="QLoRA SFT",
            four_module="no",
            calibration="base calibrated output",
            metrics=epoch_system_metrics(epoch_summary, setting, "QLoRA-M-Prometheus-3B"),
            n=3,
            source="qlora_epoch_ablation_3seed_1024_summary",
        ),
        make_row(
            system="GRM-Llama3.2-3B reward model",
            role="external 3B reward baseline",
            run_type="single full test",
            training="external reward model",
            four_module="no",
            calibration="dev-selected tie margin",
            metrics=external_system_metrics(external_report, grm_name),
            parse_failure_rate=external_parse_failure(external_report, grm_name),
            n=external_n(external_report, grm_name),
            source="external_3b_baseline_comparison_report",
        ),
        make_row(
            system="Qwen2.5-3B-Instruct",
            role="external 3B instruct baseline",
            run_type="single full test",
            training="external instruct model",
            four_module="no",
            calibration="label likelihood softmax",
            metrics=external_system_metrics(external_report, qwen_name),
            parse_failure_rate=external_parse_failure(external_report, qwen_name),
            n=external_n(external_report, qwen_name),
            source="external_3b_baseline_comparison_report",
        ),
        make_row(
            system="GLIDER",
            role="external 4B evaluator baseline",
            run_type="single full test",
            training="external evaluator model",
            four_module="no",
            calibration="rubric likelihood softmax",
            metrics=external_system_metrics(external_report, glider_name),
            parse_failure_rate=external_parse_failure(external_report, glider_name),
            n=external_n(external_report, glider_name),
            source="external_3b_baseline_comparison_report",
            notes="CC-BY-NC-4.0 license; non-commercial external evaluator baseline.",
        ),
        make_row(
            system="QLoRA-BEA-Judge epoch2",
            role="proposed accuracy-oriented model",
            run_type="3-seed mean+/-std",
            training="QLoRA SFT",
            four_module="yes",
            calibration="fusion calibration",
            metrics=epoch_system_metrics(epoch_summary, setting, "QLoRA-BEA-Judge"),
            n=3,
            source="qlora_epoch_ablation_3seed_1024_summary",
        ),
        make_row(
            system="QLoRA-BEA-Judge epoch2 + Tie rescue",
            role="proposed accuracy-constrained tie policy",
            run_type="3-seed mean+/-std",
            training="QLoRA SFT",
            four_module="yes",
            calibration="fusion calibration + dev-only tie rescue",
            metrics=rescue_metrics(tie_rescue_audit, setting),
            n=3,
            source="accuracy_constrained_tie_rescue_audit",
        ),
    ]

    proposed = rows[-2]["metrics"]
    proposed_rescue = rows[-1]["metrics"]
    grm = rows[3]["metrics"]
    qwen = rows[4]["metrics"]
    glider = rows[5]["metrics"]
    return {
        "setting": setting,
        "rows": rows,
        "key_deltas": {
            "qlora_bea_epoch2_minus_grm": {
                metric: round((metric_mean(proposed, metric) or 0.0) - (metric_mean(grm, metric) or 0.0), 6)
                for metric in METRICS
            },
            "qlora_bea_epoch2_minus_qwen": {
                metric: round((metric_mean(proposed, metric) or 0.0) - (metric_mean(qwen, metric) or 0.0), 6)
                for metric in METRICS
            },
            "qlora_bea_epoch2_minus_glider": {
                metric: round((metric_mean(proposed, metric) or 0.0) - (metric_mean(glider, metric) or 0.0), 6)
                for metric in METRICS
            },
            "tie_rescue_minus_qlora_bea_epoch2": {
                metric: round((metric_mean(proposed_rescue, metric) or 0.0) - (metric_mean(proposed, metric) or 0.0), 6)
                for metric in METRICS
            },
        },
        "recommended_claim": (
            "Under the lightweight external-baseline comparison, QLoRA-BEA-Judge epoch2 is evaluated "
            "against added 3B/4B baselines on accuracy, macro-F1, Tie recall, and calibration. The Tie-rescue "
            "variant improves Tie recall over the accuracy-oriented epoch2 operating point while "
            "preserving the accuracy target."
        ),
    }


def markdown_table(rows: List[Mapping[str, Any]]) -> str:
    fields = [
        "system",
        "role",
        "run_type",
        "accuracy",
        "macro_f1",
        "ece",
        "tie_recall",
        "parse_failure_rate",
        "n",
    ]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        metrics = row.get("metrics", {})
        values = [
            str(row.get("system", "")),
            str(row.get("role", "")),
            str(row.get("run_type", "")),
            metric_cell(metrics.get("accuracy", {})),
            metric_cell(metrics.get("macro_f1", {})),
            metric_cell(metrics.get("ece", {})),
            metric_cell(metrics.get("tie_recall", {})),
            "" if row.get("parse_failure_rate") is None else f"{float(row['parse_failure_rate']):.4f}",
            "" if row.get("n") is None else str(row.get("n")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# External Lightweight Baseline Full Comparison",
        "",
        f"Setting: `{summary['setting']}`",
        "",
        markdown_table(list(summary["rows"])).rstrip(),
        "",
        "## Key Deltas",
        "",
        "| comparison | accuracy | macro_f1 | ece | tie_recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, deltas in summary["key_deltas"].items():
        lines.append(
            f"| {name} | "
            + " | ".join(f"{float(deltas[metric]):+.4f}" for metric in METRICS)
            + " |"
        )
    lines.extend(["", "## Recommended Claim", "", str(summary["recommended_claim"]), ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build integrated external lightweight baseline comparison.")
    parser.add_argument(
        "--epoch-summary",
        default="datasets/model_outputs/qlora_epoch_ablation_3seed_1024_summary/epoch_ablation_summary.json",
    )
    parser.add_argument(
        "--external-report",
        default=f"{DEFAULT_OUTPUT_DIR}/external_3b_baseline_comparison_report.json",
    )
    parser.add_argument(
        "--tie-rescue-audit",
        default=(
            "datasets/model_outputs/accuracy_constrained_tie_rescue_global_strict_dev_summary/"
            "accuracy_constrained_tie_rescue_audit.json"
        ),
    )
    parser.add_argument("--setting", default="epoch2_1024")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        epoch_summary=load_json(resolve_root_path(args.epoch_summary)),
        external_report=load_json(resolve_root_path(args.external_report)),
        tie_rescue_audit=load_json(resolve_root_path(args.tie_rescue_audit)),
        setting=args.setting,
    )
    output_dir = resolve_root_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "external_3b_full_comparison_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "external_3b_full_comparison_table.md").write_text(
        markdown_summary(summary),
        encoding="utf-8",
    )
    print(markdown_table(list(summary["rows"])))


if __name__ == "__main__":
    main()
