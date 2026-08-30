"""Build BEA-Judge bias-awareness profiles and summary reports."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bias_awareness import (  # noqa: E402
    PAIRWISE_LABELS,
    build_bias_profile,
    dataset_accuracy_gaps,
    load_calibrated_predictions,
    summarize_bias_profiles,
)
from dataset_adapter import samples_from_payload  # noqa: E402


DATASETS = ROOT / "datasets"
DEFAULT_CALIBRATED_RESULTS = DATASETS / "model_outputs" / "bea_judge_20260510_144502" / "calibrated_results.json"
PROFILE_PATH = DATASETS / "bias_profiles.json"
REPORT_JSON = DATASETS / "bias_awareness_report.json"
REPORT_MD = DATASETS / "bias_awareness_report.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def default_dataset_files(datasets_root: Path) -> List[Path]:
    if datasets_root.is_file():
        return [datasets_root]
    return [
        datasets_root / "cleaned" / "train.json",
        datasets_root / "cleaned" / "dev.json",
        datasets_root / "cleaned" / "test.json",
        datasets_root / "cleaned_zh" / "train.json",
        datasets_root / "cleaned_zh" / "dev.json",
        datasets_root / "cleaned_zh" / "test.json",
    ]


def load_samples(paths: List[Path]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        samples.extend(samples_from_payload(payload))
    return samples


def pairwise_scope(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        sample
        for sample in samples
        if sample.get("human_label") in PAIRWISE_LABELS and sample.get("answer_b") not in {None, ""}
    ]


def prediction_coverage(samples: List[Dict[str, Any]], predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sample_ids = {str(sample.get("id")) for sample in samples}
    covered = sum(1 for sample_id in sample_ids if sample_id in predictions)
    return {
        "profile_scope_count": len(sample_ids),
        "prediction_count": len(predictions),
        "covered_count": covered,
        "coverage_ratio": round(covered / len(sample_ids), 4) if sample_ids else 0.0,
    }


def build_markdown(report: Dict[str, Any]) -> str:
    overall = report["summary"]["overall"]
    lines = [
        "# BEA-Judge Bias Awareness Report",
        "",
        f"- Created at: {report['created_at']}",
        f"- Profile scope count: {overall['profile_count']}",
        f"- Accuracy on rows with predictions: {overall['accuracy']}",
        f"- Review rate: {overall['review_rate']}",
        f"- Average bias risk: {overall['avg_overall_bias_risk']}",
        f"- Prediction coverage: {report['prediction_coverage']['coverage_ratio']}",
        "",
        "## Bias Type Summary",
        "",
    ]
    for name, row in report["summary"]["by_bias_type"].items():
        lines.append(
            f"- {name}: count={row['count']}, accuracy={row['accuracy']}, "
            f"review_rate={row['review_rate']}, avg_risk={row['avg_overall_bias_risk']}"
        )
    lines.extend(["", "## Dataset Summary", ""])
    for name, row in report["summary"]["by_dataset"].items():
        lines.append(
            f"- {name}: count={row['count']}, accuracy={row['accuracy']}, "
            f"review_rate={row['review_rate']}, avg_risk={row['avg_overall_bias_risk']}"
        )
    lines.extend(["", "## Reason Counts", ""])
    if report["summary"]["reason_counts"]:
        for name, count in report["summary"]["reason_counts"].items():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- No bias reasons emitted.")
    return "\n".join(lines) + "\n"


def run_bias_audit(
    *,
    datasets_root: Path = DATASETS,
    calibrated_results: Path | None = DEFAULT_CALIBRATED_RESULTS,
) -> Dict[str, Any]:
    dataset_files = default_dataset_files(datasets_root)
    samples = load_samples(dataset_files)
    scoped_samples = pairwise_scope(samples)
    predictions = (
        load_calibrated_predictions(calibrated_results)
        if calibrated_results is not None and calibrated_results.exists()
        else {}
    )
    gaps = dataset_accuracy_gaps(scoped_samples, predictions)
    profiles = [
        build_bias_profile(
            sample,
            predictions.get(str(sample.get("id"))),
            dataset_accuracy_gap=gaps.get(str(sample.get("dataset"))),
        )
        for sample in scoped_samples
    ]
    summary = summarize_bias_profiles(profiles)
    profile_payload = {
        "created_at": utc_now(),
        "profile_count": len(profiles),
        "calibrated_results": path_relative_to_root(calibrated_results) if calibrated_results else None,
        "profiles": profiles,
    }
    write_json(PROFILE_PATH, profile_payload)

    report = {
        "created_at": utc_now(),
        "input_files": [path_relative_to_root(path) for path in dataset_files],
        "profile_path": str(PROFILE_PATH.relative_to(ROOT)),
        "calibrated_results": path_relative_to_root(calibrated_results) if calibrated_results else None,
        "prediction_coverage": prediction_coverage(scoped_samples, predictions),
        "dataset_accuracy_gaps": gaps,
        "summary": summary,
        "validation_gates": {
            "profile_count_positive": len(profiles) > 0,
            "risk_scores_in_range": all(0.0 <= row["bias"]["overall_bias_risk"] <= 1.0 for row in profiles),
            "prediction_coverage_available": bool(predictions),
        },
    }
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BEA-Judge bias-awareness signals.")
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    parser.add_argument("--calibrated-results", type=Path, default=DEFAULT_CALIBRATED_RESULTS)
    parser.add_argument("--no-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrated = None if args.no_predictions else args.calibrated_results
    report = run_bias_audit(datasets_root=args.datasets, calibrated_results=calibrated)
    print(
        json.dumps(
            {
                "profile_count": report["summary"]["overall"]["profile_count"],
                "accuracy": report["summary"]["overall"]["accuracy"],
                "review_rate": report["summary"]["overall"]["review_rate"],
                "avg_overall_bias_risk": report["summary"]["overall"]["avg_overall_bias_risk"],
                "prediction_coverage": report["prediction_coverage"]["coverage_ratio"],
                "report": str(REPORT_JSON.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
