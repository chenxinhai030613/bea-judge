"""Build evidence-enhanced factuality profiles and summary reports."""

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

from dataset_adapter import samples_from_payload  # noqa: E402
from evidence_features import (  # noqa: E402
    build_evidence_profile,
    summarize_evidence_profiles,
)


DATASETS = ROOT / "datasets"
PROFILE_PATH = DATASETS / "evidence_profiles.json"
REPORT_JSON = DATASETS / "evidence_fact_report.json"
REPORT_MD = DATASETS / "evidence_fact_report.md"


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


def factuality_scope(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [sample for sample in samples if sample.get("task_type") == "factuality_rag"]


def _count_missing(scoped_samples: List[Dict[str, Any]], field: str) -> int:
    return sum(1 for sample in scoped_samples if sample.get(field) in {None, ""})


def build_markdown(report: Dict[str, Any]) -> str:
    overall = report["summary"]["overall"]
    lines = [
        "# BEA-Judge Evidence Factuality Report",
        "",
        f"- Created at: {report['created_at']}",
        f"- Profile count: {overall['profile_count']}",
        f"- Review rate: {overall['review_rate']}",
        f"- Average evidence risk: {overall['avg_evidence_risk']}",
        f"- Average claim support A: {overall['avg_claim_support_a']}",
        f"- Missing context rows: {report['missing_fields']['context']}",
        f"- Missing reference rows: {report['missing_fields']['reference']}",
        "",
        "## Form Summary",
        "",
    ]
    for name, row in report["summary"]["by_form"].items():
        lines.append(
            f"- {name}: count={row['count']}, review_rate={row['review_rate']}, "
            f"avg_risk={row['avg_evidence_risk']}, avg_support_a={row['avg_claim_support_a']}"
        )
    lines.extend(["", "## Dataset Summary", ""])
    for name, row in report["summary"]["by_dataset"].items():
        lines.append(
            f"- {name}: count={row['count']}, review_rate={row['review_rate']}, "
            f"avg_risk={row['avg_evidence_risk']}, avg_support_a={row['avg_claim_support_a']}"
        )
    lines.extend(["", "## Reason Counts", ""])
    if report["summary"]["reason_counts"]:
        for name, count in report["summary"]["reason_counts"].items():
            lines.append(f"- {name}: {count}")
    else:
        lines.append("- No evidence risk reasons emitted.")
    lines.extend(["", "## Validation Gates", ""])
    for name, passed in report["validation_gates"].items():
        lines.append(f"- {name}: {passed}")
    return "\n".join(lines) + "\n"


def run_evidence_audit(*, datasets_root: Path = DATASETS) -> Dict[str, Any]:
    dataset_files = default_dataset_files(datasets_root)
    samples = load_samples(dataset_files)
    scoped_samples = factuality_scope(samples)
    profiles = [build_evidence_profile(sample) for sample in scoped_samples]
    summary = summarize_evidence_profiles(profiles)
    profile_payload = {
        "created_at": utc_now(),
        "profile_count": len(profiles),
        "profiles": profiles,
    }
    write_json(PROFILE_PATH, profile_payload)

    forms = {profile.get("form") for profile in profiles}
    report = {
        "created_at": utc_now(),
        "input_files": [path_relative_to_root(path) for path in dataset_files],
        "profile_path": str(PROFILE_PATH.relative_to(ROOT)),
        "missing_fields": {
            "context": _count_missing(scoped_samples, "context"),
            "reference": _count_missing(scoped_samples, "reference"),
            "answer_a": _count_missing(scoped_samples, "answer_a"),
        },
        "summary": summary,
        "validation_gates": {
            "profile_count_positive": len(profiles) > 0,
            "risk_scores_in_range": all(0.0 <= row["evidence"]["evidence_risk"] <= 1.0 for row in profiles),
            "contains_single_answer": "single_answer" in forms,
            "contains_pairwise": "pairwise" in forms,
            "context_present_for_all": _count_missing(scoped_samples, "context") == 0,
            "reference_present_for_all": _count_missing(scoped_samples, "reference") == 0,
        },
    }
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(build_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit BEA-Judge evidence-enhanced factuality signals.")
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_evidence_audit(datasets_root=args.datasets)
    print(
        json.dumps(
            {
                "profile_count": report["summary"]["overall"]["profile_count"],
                "review_rate": report["summary"]["overall"]["review_rate"],
                "avg_evidence_risk": report["summary"]["overall"]["avg_evidence_risk"],
                "avg_claim_support_a": report["summary"]["overall"]["avg_claim_support_a"],
                "report": str(REPORT_JSON.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
