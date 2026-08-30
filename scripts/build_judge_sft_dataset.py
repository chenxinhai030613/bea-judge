"""Build pairwise SFT JSONL files for M-Prometheus QLoRA training."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from base_judge import build_prometheus_pairwise_prompt  # noqa: E402
from dataset_adapter import samples_from_payload  # noqa: E402


PAIRWISE_TARGETS = {
    "A>B": "[RESULT] A",
    "B>A": "[RESULT] B",
    "Tie": "[RESULT] Tie",
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def pairwise_sft_rows(
    samples: Iterable[Dict[str, Any]],
    *,
    split: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        if sample.get("split") != split:
            continue
        label = sample.get("human_label")
        if label not in PAIRWISE_TARGETS:
            continue
        if not str(sample.get("answer_b") or "").strip():
            continue
        rows.append(
            {
                "id": sample.get("id"),
                "split": split,
                "dataset": sample.get("dataset"),
                "task_type": sample.get("task_type"),
                "human_label": label,
                "prompt_text": build_prometheus_pairwise_prompt(sample),
                "target_text": PAIRWISE_TARGETS[str(label)],
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset(
    *,
    input_path: Path,
    output_dir: Path,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
) -> Dict[str, Any]:
    samples = samples_from_payload(load_json(input_path))
    test_pairwise = [
        sample
        for sample in samples
        if sample.get("split") == "test"
        and sample.get("human_label") in PAIRWISE_TARGETS
        and str(sample.get("answer_b") or "").strip()
    ]
    train_rows = pairwise_sft_rows(samples, split="train", max_samples=max_train_samples)
    dev_rows = pairwise_sft_rows(samples, split="dev", max_samples=max_dev_samples)
    if any(row["split"] == "test" for row in train_rows + dev_rows):
        raise ValueError("test rows leaked into SFT export")

    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(dev_path, dev_rows)
    metadata = {
        "input_dataset": str(input_path),
        "output_dir": str(output_dir),
        "train_jsonl": str(train_path),
        "dev_jsonl": str(dev_path),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "heldout_test_pairwise_rows": len(test_pairwise),
        "target_distribution": {
            "train": dict(Counter(row["target_text"] for row in train_rows)),
            "dev": dict(Counter(row["target_text"] for row in dev_rows)),
        },
        "target_schema": sorted(set(PAIRWISE_TARGETS.values())),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pairwise judge SFT JSONL files.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "qlora_judge_sft.json"))
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None, help="Accepted for runbook compatibility.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(resolve_root_path(args.config)) if args.config else {}
    data_config = config.get("data", {})
    input_path = resolve_root_path(args.input or data_config.get("input_dataset"))
    output_dir = resolve_root_path(args.output_dir or data_config.get("output_dir"))
    metadata = build_dataset(
        input_path=input_path,
        output_dir=output_dir,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
