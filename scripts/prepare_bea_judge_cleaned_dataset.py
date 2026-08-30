"""Combine cleaned BEA-Judge splits into one training/scoring payload."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_adapter import samples_from_payload  # noqa: E402


DATASETS = ROOT / "datasets"
DEFAULT_OUTPUT = DATASETS / "processed" / "bea_judge_cleaned_3400.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_input_files(datasets_root: Path = DATASETS) -> List[Path]:
    return [
        datasets_root / "cleaned" / "train.json",
        datasets_root / "cleaned" / "dev.json",
        datasets_root / "cleaned" / "test.json",
        datasets_root / "cleaned_zh" / "train.json",
        datasets_root / "cleaned_zh" / "dev.json",
        datasets_root / "cleaned_zh" / "test.json",
    ]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cleaned_samples(paths: List[Path]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in paths:
        samples.extend(samples_from_payload(read_json(path)))
    return samples


def build_payload(samples: List[Dict[str, Any]], input_files: List[Path]) -> Dict[str, Any]:
    return {
        "dataset_info": {
            "name": "bea_judge_cleaned_combined",
            "created_at": utc_now(),
            "schema": "BEA-Judge flat training schema adapted from canonical cleaned splits",
            "input_files": [str(path.relative_to(ROOT)) for path in input_files],
            "sample_count": len(samples),
            "by_split": dict(Counter(str(sample.get("split")) for sample in samples)),
            "by_task_type": dict(Counter(str(sample.get("task_type")) for sample in samples)),
            "by_language": dict(Counter(str(sample.get("language")) for sample in samples)),
        },
        "samples": samples,
    }


def write_combined_dataset(output: Path, payload: Dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare combined BEA-Judge cleaned dataset.")
    parser.add_argument("--datasets", type=Path, default=DATASETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = default_input_files(args.datasets)
    samples = load_cleaned_samples(input_files)
    payload = build_payload(samples, input_files)
    write_combined_dataset(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sample_count": payload["dataset_info"]["sample_count"],
                "by_task_type": payload["dataset_info"]["by_task_type"],
                "by_split": payload["dataset_info"]["by_split"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
