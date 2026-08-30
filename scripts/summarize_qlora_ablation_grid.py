"""Summarize QLoRA epoch/data ablation reports across seeds and settings."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
SYSTEMS = ("Raw M-Prometheus-3B", "Current BEA-Judge", "QLoRA-M-Prometheus-3B", "QLoRA-BEA-Judge")
METRICS = ("accuracy", "macro_f1", "ece", "tie_recall")


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(float(statistics.fmean(values)), 6),
        "std": round(float(statistics.stdev(values)), 6) if len(values) > 1 else 0.0,
        "min": round(float(min(values)), 6),
        "max": round(float(max(values)), 6),
    }


def metric_values(rows: Sequence[Mapping[str, Any]], system: str, metric: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if row.get("system") != system:
            continue
        value = row.get(metric)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def load_setting_report(setting: str, seeds: Sequence[str], template: str) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    paths: List[str] = []
    for seed in seeds:
        path = resolve_root_path(template.format(setting=setting, seed=seed))
        report = load_json(path)
        rows = list(report.get("comparison_rows", []))
        for row in rows:
            enriched = dict(row)
            enriched["seed"] = seed
            enriched["setting"] = setting
            all_rows.append(enriched)
        paths.append(str(path))
    by_system = {
        system: {metric: summarize(metric_values(all_rows, system, metric)) for metric in METRICS}
        for system in SYSTEMS
    }
    return {"setting": setting, "sources": paths, "by_system": by_system, "per_seed_rows": all_rows}


def build_summary(settings: Sequence[str], seeds: Sequence[str], template: str) -> Dict[str, Any]:
    return {
        "settings": list(settings),
        "seeds": list(seeds),
        "std_type": "sample",
        "results": [load_setting_report(setting, seeds, template) for setting in settings],
    }


def format_mean_std(item: Mapping[str, Optional[float]]) -> str:
    if item.get("mean") is None:
        return ""
    return f"{float(item['mean']):.4f} +/- {float(item['std'] or 0.0):.4f}"


def markdown_summary(summary: Mapping[str, Any], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        f"Settings: {', '.join(summary['settings'])}",
        f"Seeds: {', '.join(summary['seeds'])}",
        "Std type: sample",
        "",
    ]
    for result in summary["results"]:
        lines.extend(
            [
                f"## {result['setting']}",
                "",
                "| system | accuracy | macro_f1 | ece | tie_recall |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for system in SYSTEMS:
            metrics = result["by_system"][system]
            lines.append(
                "| "
                + " | ".join(
                    [
                        system,
                        format_mean_std(metrics["accuracy"]),
                        format_mean_std(metrics["macro_f1"]),
                        format_mean_std(metrics["ece"]),
                        format_mean_std(metrics["tie_recall"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize QLoRA epoch/data ablation reports.")
    parser.add_argument("--settings", nargs="*", required=True)
    parser.add_argument("--seeds", nargs="*", default=["13", "42", "2026"])
    parser.add_argument("--report-template", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(
        settings=[str(item) for item in args.settings],
        seeds=[str(item) for item in args.seeds],
        template=str(args.report_template),
    )
    output_json = resolve_root_path(args.output_json)
    output_md = resolve_root_path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(markdown_summary(summary, str(args.title)), encoding="utf-8")
    print(json.dumps({"settings": summary["settings"], "seeds": summary["seeds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
