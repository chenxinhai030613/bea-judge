"""Build deterministic stratified SFT subsets for QLoRA data ablation."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
DEFAULT_SOURCE_DIR = ROOT / "datasets" / "sft" / "m_prometheus_pairwise"
DEFAULT_OUTPUT_ROOT = ROOT / "datasets" / "sft"


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def stratum_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('target_text','')}||{row.get('dataset','')}"


def stratified_sample_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> Dict[str, int]:
    if sample_size <= 0 or sample_size > len(rows):
        raise ValueError(f"sample_size must be in [1, {len(rows)}], got {sample_size}")
    by_stratum: MutableMapping[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_stratum[stratum_key(row)].append(idx)

    exact: Dict[str, float] = {}
    base: Dict[str, int] = {}
    for key, indices in by_stratum.items():
        target = sample_size * len(indices) / len(rows)
        exact[key] = target
        base[key] = int(target)

    assigned = sum(base.values())
    remaining = sample_size - assigned
    if remaining > 0:
        rng = random.Random(seed)
        ranked = sorted(
            by_stratum.keys(),
            key=lambda key: (-(exact[key] - base[key]), rng.random()),
        )
        for key in ranked[:remaining]:
            base[key] += 1
    return base


def build_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    counts = stratified_sample_counts(rows, sample_size=sample_size, seed=seed)
    rng = random.Random(seed)
    by_stratum: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[stratum_key(row)].append(dict(row))

    selected: List[Dict[str, Any]] = []
    for key in sorted(by_stratum.keys()):
        bucket = by_stratum[key]
        rng.shuffle(bucket)
        selected.extend(bucket[: counts.get(key, 0)])
    selected.sort(key=lambda row: str(row.get("id")))
    if len(selected) != sample_size:
        raise ValueError(f"built subset size mismatch: expected {sample_size}, got {len(selected)}")
    return selected


def metadata_for_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "rows": len(rows),
        "target_distribution": dict(Counter(str(row.get("target_text")) for row in rows)),
        "dataset_distribution": dict(Counter(str(row.get("dataset")) for row in rows)),
    }


def build_subsets(
    *,
    source_dir: Path,
    output_root: Path,
    sample_sizes: Sequence[int],
    seed: int,
) -> Dict[str, Any]:
    train_rows = load_jsonl(source_dir / "train.jsonl")
    dev_rows = load_jsonl(source_dir / "dev.jsonl")
    summary: Dict[str, Any] = {
        "source_dir": str(source_dir),
        "subset_seed": seed,
        "train_rows_full": len(train_rows),
        "dev_rows_full": len(dev_rows),
        "subsets": [],
    }
    for sample_size in sample_sizes:
        fraction = sample_size / len(train_rows)
        tag = f"sft{int(round(fraction * 100)):02d}"
        subset_dir = output_root / f"m_prometheus_pairwise_{tag}"
        subset_train = build_subset(train_rows, sample_size=sample_size, seed=seed)
        write_jsonl(subset_dir / "train.jsonl", subset_train)
        write_jsonl(subset_dir / "dev.jsonl", dev_rows)
        metadata = {
            "source_dir": str(source_dir),
            "subset_seed": seed,
            "sample_size": sample_size,
            "fraction": round(fraction, 6),
            "train": metadata_for_rows(subset_train),
            "dev": metadata_for_rows(dev_rows),
        }
        (subset_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["subsets"].append({"tag": tag, "dir": str(subset_dir), **metadata})
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic stratified QLoRA SFT subsets.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--sample-sizes", nargs="*", type=int, default=[1202, 2403])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_subsets(
        source_dir=resolve_root_path(args.source_dir),
        output_root=resolve_root_path(args.output_root),
        sample_sizes=[int(value) for value in args.sample_sizes],
        seed=int(args.seed),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
