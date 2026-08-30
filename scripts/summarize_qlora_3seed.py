"""Summarize QLoRA epoch1 comparison reports across seeds."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
NUMERIC_METRICS = ("accuracy", "macro_f1", "ece", "tie_recall")


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def seed_from_path(path: Path, fallback: str) -> str:
    match = re.search(r"seed(\d+)", str(path))
    return match.group(1) if match else fallback


def load_report(path: Path, *, seed: str) -> Dict[str, Any]:
    report_path = path / "qlora_comparison_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = report.get("gate", {})
    return {
        "seed": seed,
        "path": str(path),
        "gate_passed": bool(gate.get("passed", False)),
        "checks": dict(gate.get("checks", {})),
        "comparison_rows": list(report.get("comparison_rows", [])),
        "raw_qlora_metrics": dict(report.get("raw_qlora_metrics", {})),
    }


def metric_values(rows: Iterable[Mapping[str, Any]], system: str, metric: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        if row.get("system") != system:
            continue
        value = row.get(metric)
        if value in ("", None):
            continue
        values.append(float(value))
    return values


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


def build_summary(reports: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    systems: List[str] = []
    for report in reports:
        for row in report["comparison_rows"]:
            row_with_seed = dict(row)
            row_with_seed["seed"] = report["seed"]
            all_rows.append(row_with_seed)
            system = str(row.get("system"))
            if system not in systems:
                systems.append(system)

    by_system: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for system in systems:
        by_system[system] = {
            metric: summarize_values(metric_values(all_rows, system, metric)) for metric in NUMERIC_METRICS
        }

    return {
        "seeds": [report["seed"] for report in reports],
        "std_type": "sample",
        "gate": {
            "passed_count": sum(1 for report in reports if report["gate_passed"]),
            "total": len(reports),
            "all_passed": all(report["gate_passed"] for report in reports) if reports else False,
            "per_seed": {report["seed"]: report["gate_passed"] for report in reports},
        },
        "by_system": by_system,
        "per_seed_rows": all_rows,
    }


def format_mean_std(summary: Mapping[str, Optional[float]]) -> str:
    if summary.get("mean") is None:
        return ""
    return f"{summary['mean']:.4f} +/- {summary['std']:.4f}"


def markdown_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "# QLoRA Epoch1 Three-Seed Summary",
        "",
        f"Seeds: {', '.join(summary['seeds'])}",
        f"Gate passed: {summary['gate']['passed_count']}/{summary['gate']['total']}",
        "Std type: sample",
        "",
        "| system | accuracy | macro_f1 | ece | tie_recall |",
        "| --- | --- | --- | --- | --- |",
    ]
    for system, metrics in summary["by_system"].items():
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
    lines.extend(["", "## Per-Seed Rows", ""])
    if summary["per_seed_rows"]:
        fields = ["seed", "system", "base", "four_module", "accuracy", "macro_f1", "ece", "tie_recall"]
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in summary["per_seed_rows"]:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "three_seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "three_seed_summary.md").write_text(markdown_summary(summary), encoding="utf-8")


def comparison_dirs_from_args(args: argparse.Namespace) -> List[Path]:
    if args.comparison_dirs:
        return [resolve_root_path(path) for path in args.comparison_dirs]
    input_root = resolve_root_path(args.input_root)
    run_suffix = getattr(args, "run_suffix", "_1024")
    dirs: List[Path] = []
    for seed in args.seeds:
        preferred = input_root / f"qlora_comparison_seed{seed}_epoch1{run_suffix}"
        legacy = input_root / f"qlora_comparison_seed{seed}_epoch1"
        if preferred.exists():
            dirs.append(preferred)
        elif legacy.exists():
            dirs.append(legacy)
        else:
            dirs.append(preferred)
    return dirs


def output_dir_from_args(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return resolve_root_path(args.output_dir)
    return resolve_root_path(
        f"datasets/model_outputs/qlora_3seed_epoch1{args.run_suffix}_summary"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize QLoRA epoch1 reports across seeds.")
    parser.add_argument("--seeds", nargs="*", default=["13", "42", "2026"])
    parser.add_argument("--comparison-dirs", nargs="*", default=None)
    parser.add_argument("--input-root", default="datasets/model_outputs")
    parser.add_argument(
        "--run-suffix",
        default="_1024",
        help="Suffix used by epoch1 comparison directories when --comparison-dirs is not provided.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: List[Dict[str, Any]] = []
    for index, path in enumerate(comparison_dirs_from_args(args)):
        seed = seed_from_path(path, args.seeds[index] if index < len(args.seeds) else str(index))
        try:
            reports.append(load_report(path, seed=seed))
        except FileNotFoundError:
            if not args.allow_missing:
                raise
            print(f"missing comparison report for seed {seed}: {path}")
    summary = build_summary(reports)
    write_summary(summary, output_dir_from_args(args))
    print(json.dumps(summary["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
