"""Summarize QLoRA four-module ablation reports across seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
NUMERIC_METRICS = (
    "accuracy",
    "macro_f1",
    "ece",
    "brier",
    "tie_recall",
    "review_rate",
    "review_capture_rate",
)


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 6),
        "std": round(float(statistics.stdev(values)), 6) if len(values) > 1 else 0.0,
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def iter_metric_entries(report: Mapping[str, Any]) -> Iterable[Tuple[str, str, str, Mapping[str, Any]]]:
    for section in ("variants", "control_baselines"):
        for variant in report.get(section, []):
            name = str(variant.get("name", ""))
            if not name:
                continue
            for head in ("pairwise", "factuality"):
                metrics = variant.get(head, {}).get("test_metrics")
                if isinstance(metrics, Mapping):
                    yield section, name, head, metrics
    for variant in report.get("feature_group_ablations", []):
        name = str(variant.get("name", ""))
        metrics = variant.get("test_metrics")
        if name and isinstance(metrics, Mapping):
            yield "feature_group_ablations", name, "factuality", metrics
    for row in report.get("bias_utility", []):
        name = str(row.get("setting", ""))
        if name:
            yield "bias_utility", name, str(row.get("head", "pairwise")), row


def build_summary(seed_reports: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    collected: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {}
    for item in seed_reports:
        seed = str(item["seed"])
        for section, name, head, metrics in iter_metric_entries(item["report"]):
            bucket = collected.setdefault(section, {}).setdefault(name, {}).setdefault(head, {})
            for metric in NUMERIC_METRICS:
                value = as_float(metrics.get(metric))
                if value is not None:
                    bucket.setdefault(metric, []).append(value)
            bucket.setdefault("_seeds", []).append(seed)

    sections: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for section, by_name in collected.items():
        sections[section] = {}
        for name, by_head in by_name.items():
            sections[section][name] = {}
            for head, values_by_metric in by_head.items():
                seeds = [str(seed) for seed in values_by_metric.get("_seeds", [])]
                sections[section][name][head] = {
                    "seeds": seeds,
                    "metrics": {
                        metric: summarize_values(values_by_metric.get(metric, [])) for metric in NUMERIC_METRICS
                    },
                }
    return {
        "seeds": [str(item["seed"]) for item in seed_reports],
        "std_type": "sample",
        "sources": [str(item["path"]) for item in seed_reports],
        "sections": sections,
    }


def format_mean_std(summary: Mapping[str, Optional[float]]) -> str:
    if summary.get("mean") is None:
        return ""
    return f"{float(summary['mean']):.4f} +/- {float(summary['std'] or 0.0):.4f}"


def markdown_summary(summary: Mapping[str, Any]) -> str:
    lines = [
        "# QLoRA Four-Module Ablation Three-Seed Summary",
        "",
        f"Seeds: {', '.join(summary['seeds'])}",
        "Std type: sample",
        "",
    ]
    for section, by_name in summary["sections"].items():
        lines.extend(
            [
                f"## {section}",
                "",
                "| name | head | accuracy | macro_f1 | ece | brier | tie_recall | review_rate | review_capture_rate |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, by_head in by_name.items():
            for head, payload in by_head.items():
                metrics = payload["metrics"]
                lines.append(
                    f"| {name} | {head} | "
                    + " | ".join(format_mean_std(metrics[metric]) for metric in NUMERIC_METRICS)
                    + " |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_seed_reports(
    *,
    seeds: Sequence[str],
    report_template: str,
    allow_missing: bool,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for seed in seeds:
        path = resolve_root_path(report_template.format(seed=seed))
        if not path.exists():
            if allow_missing:
                print(f"missing ablation report for seed {seed}: {path}")
                continue
            raise FileNotFoundError(path)
        reports.append({"seed": seed, "path": path, "report": load_json(path)})
    return reports


def write_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_3seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "ablation_3seed_summary.md").write_text(markdown_summary(summary), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize QLoRA ablation reports across seeds.")
    parser.add_argument("--seeds", nargs="*", default=["13", "42", "2026"])
    parser.add_argument(
        "--report-template",
        default="datasets/model_outputs/qlora_ablation_seed{seed}_epoch1_1024/ablation_report.json",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/model_outputs/qlora_ablation_3seed_epoch1_1024_summary",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = load_seed_reports(
        seeds=[str(seed) for seed in args.seeds],
        report_template=args.report_template,
        allow_missing=bool(args.allow_missing),
    )
    summary = build_summary(reports)
    write_summary(summary, resolve_root_path(args.output_dir))
    print(json.dumps({"seeds": summary["seeds"], "sources": summary["sources"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
