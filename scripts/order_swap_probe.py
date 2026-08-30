"""Dev-only order-swap diagnostics for BEA-Judge pairwise samples."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / "datasets"
DEFAULT_INPUT = DATASETS / "processed" / "bea_judge_cleaned_3400.json"
DEFAULT_BASE_SCORES = (
    DATASETS
    / "judge_outputs"
    / "m_prometheus_3b_real_full_promptfix_256"
    / "base_scores.repaired.json"
)
DEFAULT_OUTPUT_DIR = DATASETS / "judge_outputs" / "order_swap_probe"
PAIRWISE_LABELS = {"A>B", "B>A", "Tie"}
DEFAULT_TARGET_DATASETS = ("mt_bench", "pandalm", "judgebench")

sys.path.insert(0, str(ROOT / "src"))

from base_judge import JudgeConfig, base_score_rows_for_disk, evaluate_samples, make_backend  # noqa: E402
from dataset_adapter import samples_from_payload  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def path_relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_latest_calibrated_results(config_path: Path) -> Path:
    payload = load_json(config_path)
    calibrated = payload.get("latest_outputs", {}).get("calibrated_results")
    if not calibrated:
        raise ValueError(f"missing latest calibrated_results in {config_path}")
    return ROOT / calibrated


def read_samples(path: Path) -> List[Dict[str, Any]]:
    return samples_from_payload(load_json(path))


def invert_pairwise_label(label: Any) -> Optional[str]:
    mapping = {"A>B": "B>A", "B>A": "A>B", "Tie": "Tie"}
    return mapping.get(str(label))


def base_score_map(base_scores: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in base_scores:
        if not isinstance(row, dict):
            continue
        scores = row.get("parsed_scores", {})
        if (
            row.get("id") is not None
            and row.get("pred_label") in PAIRWISE_LABELS
            and isinstance(scores.get("score_a"), (int, float))
            and isinstance(scores.get("score_b"), (int, float))
            and row.get("parse_status") not in {"failed", "backend_error"}
        ):
            out[str(row["id"])] = row
    return out


def calibrated_pairwise_rows(calibrated: Dict[str, Any], split: str = "dev") -> List[Dict[str, Any]]:
    return list(calibrated.get(split, {}).get("pairwise", []))


def score_margin(row: Dict[str, Any]) -> float:
    scores = row.get("parsed_scores", {})
    if isinstance(scores.get("score_a"), (int, float)) and isinstance(scores.get("score_b"), (int, float)):
        return abs(float(scores["score_a"]) - float(scores["score_b"]))
    return 0.0


def selection_reasons(
    calibrated_row: Dict[str, Any],
    base_row: Dict[str, Any],
    *,
    low_confidence_threshold: float,
) -> List[str]:
    reasons: List[str] = []
    gold = calibrated_row.get("human_label")
    calibrated_pred = calibrated_row.get("predicted_label")
    base_pred = base_row.get("pred_label")
    confidence = float(calibrated_row.get("confidence", 0.0))
    if gold in PAIRWISE_LABELS and calibrated_pred in PAIRWISE_LABELS and gold != calibrated_pred:
        reasons.append("calibrated_error")
    if confidence <= low_confidence_threshold:
        reasons.append("low_confidence")
    if base_pred in PAIRWISE_LABELS and calibrated_pred in PAIRWISE_LABELS and base_pred != calibrated_pred:
        reasons.append("base_calibrated_disagreement")
    if gold == "Tie" or calibrated_pred == "Tie" or base_pred == "Tie":
        reasons.append("tie_case")
    return reasons


def select_probe_entries(
    *,
    samples: Sequence[Dict[str, Any]],
    calibrated: Dict[str, Any],
    base_scores: Sequence[Dict[str, Any]],
    target_datasets: Sequence[str] = DEFAULT_TARGET_DATASETS,
    low_confidence_threshold: float = 0.70,
    per_dataset_limit: int = 20,
) -> List[Dict[str, Any]]:
    sample_by_id = {str(sample.get("id")): sample for sample in samples}
    base_by_id = base_score_map(base_scores)
    targets = set(target_datasets)
    candidates: List[Dict[str, Any]] = []
    for calibrated_row in calibrated_pairwise_rows(calibrated, "dev"):
        sample_id = str(calibrated_row.get("id"))
        sample = sample_by_id.get(sample_id)
        base_row = base_by_id.get(sample_id)
        if sample is None or base_row is None:
            continue
        if sample.get("dataset") not in targets or sample.get("split") != "dev":
            continue
        if sample.get("human_label") not in PAIRWISE_LABELS or not sample.get("answer_b"):
            continue
        reasons = selection_reasons(
            calibrated_row,
            base_row,
            low_confidence_threshold=low_confidence_threshold,
        )
        if not reasons:
            continue
        priority = (
            0 if "calibrated_error" in reasons else 1,
            0 if "low_confidence" in reasons else 1,
            float(calibrated_row.get("confidence", 0.0)),
            str(sample.get("dataset")),
            sample_id,
        )
        candidates.append(
            {
                "id": sample_id,
                "sample": sample,
                "base": base_row,
                "calibrated": calibrated_row,
                "selection_reasons": reasons,
                "priority": priority,
            }
        )

    candidates.sort(key=lambda row: row["priority"])
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = defaultdict(int)
    for entry in candidates:
        dataset = str(entry["sample"].get("dataset"))
        if counts[dataset] >= per_dataset_limit:
            continue
        counts[dataset] += 1
        selected.append({key: value for key, value in entry.items() if key != "priority"})
    return selected


def swapped_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    swapped = dict(sample)
    swapped["answer_a"], swapped["answer_b"] = sample.get("answer_b"), sample.get("answer_a")
    swapped["human_label"] = invert_pairwise_label(sample.get("human_label"))
    metadata = dict(sample.get("metadata", {}))
    metadata["order_swap_probe"] = {
        "original_id": sample.get("id"),
        "original_human_label": sample.get("human_label"),
    }
    swapped["metadata"] = metadata
    return swapped


def annotate_swap_rows(
    rows: Sequence[Dict[str, Any]],
    entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entry_by_id = {str(entry["id"]): entry for entry in entries}
    annotated: List[Dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        entry = entry_by_id.get(str(row.get("id")))
        if entry:
            original = entry["sample"]
            calibrated = entry["calibrated"]
            out["original_id"] = original.get("id")
            out["original_dataset"] = original.get("dataset")
            out["original_human_label"] = original.get("human_label")
            out["swapped_human_label"] = invert_pairwise_label(original.get("human_label"))
            out["original_calibrated_label"] = calibrated.get("predicted_label")
            out["selection_reasons"] = entry.get("selection_reasons", [])
        annotated.append(out)
    return annotated


def run_swap_scoring(
    entries: Sequence[Dict[str, Any]],
    *,
    backend: str,
    model_path: str,
    device: str,
    max_new_tokens: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config = JudgeConfig(
        name="order_swap_probe",
        version=model_path,
        backend=backend,
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        device=device,
        allow_fallback=False,
    )
    backend_impl = make_backend(config)
    swapped = [swapped_sample(entry["sample"]) for entry in entries]
    summary, valid_rows = evaluate_samples(swapped, config, backend=backend_impl)
    disk_rows = base_score_rows_for_disk(valid_rows, summary["parse_failures"])
    return summary, annotate_swap_rows(disk_rows, entries)


def build_consistency_rows(
    entries: Sequence[Dict[str, Any]],
    swap_scores: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    swap_by_id = {str(row.get("id")): row for row in swap_scores if isinstance(row, dict)}
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        sample = entry["sample"]
        base = entry["base"]
        calibrated = entry["calibrated"]
        sample_id = str(entry["id"])
        swap = swap_by_id.get(sample_id, {})
        swap_pred = swap.get("pred_label")
        base_pred = base.get("pred_label")
        expected_swap = invert_pairwise_label(base_pred)
        swap_available = 1.0 if swap_pred in PAIRWISE_LABELS else 0.0
        consistent = 1.0 if swap_available and swap_pred == expected_swap else 0.0
        margin_delta = abs(score_margin(base) - score_margin(swap)) if swap_available else 0.0
        calibrated_error = calibrated.get("predicted_label") != calibrated.get("human_label")
        rows.append(
            {
                "id": sample_id,
                "dataset": sample.get("dataset"),
                "task_type": sample.get("task_type"),
                "human_label": sample.get("human_label"),
                "base_pred_label": base_pred,
                "calibrated_pred_label": calibrated.get("predicted_label"),
                "confidence": round(float(calibrated.get("confidence", 0.0)), 6),
                "selection_reasons": ",".join(entry.get("selection_reasons", [])),
                "original_base_margin": round(score_margin(base), 6),
                "swap_pred_label": swap_pred,
                "expected_swap_label": expected_swap,
                "swap_available": swap_available,
                "swap_consistency_flag": consistent,
                "swap_margin_delta": round(margin_delta, 6),
                "calibrated_error": bool(calibrated_error),
                "base_error": base_pred != sample.get("human_label"),
                "tie_case": sample.get("human_label") == "Tie"
                or base_pred == "Tie"
                or calibrated.get("predicted_label") == "Tie",
            }
        )
    return rows


def summarize_consistency(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[str(row.get("dataset", "unknown"))].append(row)

    summaries: List[Dict[str, Any]] = []
    for dataset in sorted(groups, key=lambda name: (name != "overall", name)):
        items = groups[dataset]
        available = [row for row in items if float(row.get("swap_available", 0.0)) == 1.0]
        inconsistent = [row for row in available if float(row.get("swap_consistency_flag", 0.0)) == 0.0]
        errors = [row for row in items if row.get("calibrated_error")]
        inconsistent_errors = [row for row in inconsistent if row.get("calibrated_error")]
        summaries.append(
            {
                "dataset": dataset,
                "selected_n": len(items),
                "swap_available_n": len(available),
                "swap_parse_success_rate": round(len(available) / len(items), 4) if items else 0.0,
                "swap_consistency_rate": round(
                    sum(float(row.get("swap_consistency_flag", 0.0)) for row in available) / len(available),
                    4,
                )
                if available
                else "",
                "swap_inconsistency_rate": round(len(inconsistent) / len(available), 4) if available else "",
                "calibrated_error_rate": round(len(errors) / len(items), 4) if items else 0.0,
                "error_rate_when_inconsistent": round(len(inconsistent_errors) / len(inconsistent), 4)
                if inconsistent
                else "",
                "avg_swap_margin_delta": round(
                    sum(float(row.get("swap_margin_delta", 0.0)) for row in available) / len(available),
                    4,
                )
                if available
                else "",
                "tie_case_rate": round(sum(1 for row in items if row.get("tie_case")) / len(items), 4)
                if items
                else 0.0,
            }
        )
    return summaries


def hard_example_rows(entries: Sequence[Dict[str, Any]], consistency_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    consistency_by_id = {str(row["id"]): row for row in consistency_rows}
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        sample = entry["sample"]
        calibrated = entry["calibrated"]
        base = entry["base"]
        consistency = consistency_by_id.get(str(entry["id"]), {})
        answer_a = str(sample.get("answer_a") or "")
        answer_b = str(sample.get("answer_b") or "")
        rows.append(
            {
                "id": entry["id"],
                "dataset": sample.get("dataset"),
                "human_label": sample.get("human_label"),
                "base_pred_label": base.get("pred_label"),
                "calibrated_pred_label": calibrated.get("predicted_label"),
                "confidence": round(float(calibrated.get("confidence", 0.0)), 6),
                "selection_reasons": ",".join(entry.get("selection_reasons", [])),
                "base_margin": round(score_margin(base), 6),
                "answer_len_diff": len(answer_a) - len(answer_b),
                "prompt_chars": len(str(sample.get("prompt") or "")),
                "swap_pred_label": consistency.get("swap_pred_label", ""),
                "swap_consistency_flag": consistency.get("swap_consistency_flag", 0.0),
                "swap_margin_delta": consistency.get("swap_margin_delta", 0.0),
            }
        )
    return rows


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(row.get(column, "")) for column in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_table(output_dir: Path, stem: str, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> Dict[str, str]:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    write_csv(csv_path, rows, columns)
    md_path.write_text(markdown_table(rows, columns), encoding="utf-8")
    return {"csv": path_relative_to_root(csv_path), "md": path_relative_to_root(md_path)}


def run_probe(
    *,
    input_path: Path,
    calibrated_results_path: Path,
    base_scores_path: Path,
    output_dir: Path,
    target_datasets: Sequence[str],
    low_confidence_threshold: float,
    per_dataset_limit: int,
    dry_run: bool,
    backend: str,
    model_path: str,
    device: str,
    max_new_tokens: int,
    existing_swap_scores_path: Optional[Path] = None,
) -> Dict[str, Any]:
    samples = read_samples(input_path)
    calibrated = load_json(calibrated_results_path)
    base_scores = load_json(base_scores_path)
    if not isinstance(base_scores, list):
        raise ValueError(f"base scores must be a list: {base_scores_path}")
    entries = select_probe_entries(
        samples=samples,
        calibrated=calibrated,
        base_scores=base_scores,
        target_datasets=target_datasets,
        low_confidence_threshold=low_confidence_threshold,
        per_dataset_limit=per_dataset_limit,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_samples_path = output_dir / "selected_swap_samples.json"
    selected_samples_path.write_text(
        json.dumps(
            [
                {
                    "id": entry["id"],
                    "dataset": entry["sample"].get("dataset"),
                    "task_type": entry["sample"].get("task_type"),
                    "human_label": entry["sample"].get("human_label"),
                    "base_pred_label": entry["base"].get("pred_label"),
                    "calibrated_pred_label": entry["calibrated"].get("predicted_label"),
                    "confidence": entry["calibrated"].get("confidence"),
                    "selection_reasons": entry["selection_reasons"],
                }
                for entry in entries
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    scoring_summary: Dict[str, Any] = {
        "attempted": False,
        "dry_run": dry_run,
        "backend": backend,
        "model_path": model_path,
        "max_new_tokens": max_new_tokens,
        "selected_count": len(entries),
    }
    swap_scores: List[Dict[str, Any]] = []
    if existing_swap_scores_path is not None:
        swap_scores = load_json(existing_swap_scores_path)
        scoring_summary.update(
            {
                "attempted": True,
                "used_existing_swap_scores": path_relative_to_root(existing_swap_scores_path),
                "score_row_count": len(swap_scores),
            }
        )
    elif not dry_run and entries:
        summary, swap_scores = run_swap_scoring(
            entries,
            backend=backend,
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        swap_scores_path = output_dir / "swap_scores.json"
        swap_scores_path.write_text(json.dumps(swap_scores, ensure_ascii=False, indent=2), encoding="utf-8")
        scoring_summary.update(
            {
                "attempted": True,
                "score_row_count": len(swap_scores),
                "parse_failure_count": int(summary.get("parse_failure_count", 0)),
                "swap_scores": path_relative_to_root(swap_scores_path),
            }
        )

    consistency = build_consistency_rows(entries, swap_scores)
    dataset_summary = summarize_consistency(consistency)
    examples = hard_example_rows(entries, consistency)
    consistency_columns = [
        "id",
        "dataset",
        "human_label",
        "base_pred_label",
        "calibrated_pred_label",
        "confidence",
        "selection_reasons",
        "swap_pred_label",
        "expected_swap_label",
        "swap_available",
        "swap_consistency_flag",
        "swap_margin_delta",
        "calibrated_error",
    ]
    summary_columns = [
        "dataset",
        "selected_n",
        "swap_available_n",
        "swap_parse_success_rate",
        "swap_consistency_rate",
        "swap_inconsistency_rate",
        "calibrated_error_rate",
        "error_rate_when_inconsistent",
        "avg_swap_margin_delta",
        "tie_case_rate",
    ]
    example_columns = [
        "id",
        "dataset",
        "human_label",
        "base_pred_label",
        "calibrated_pred_label",
        "confidence",
        "selection_reasons",
        "base_margin",
        "answer_len_diff",
        "prompt_chars",
        "swap_pred_label",
        "swap_consistency_flag",
        "swap_margin_delta",
    ]
    tables = {
        "swap_consistency": write_table(output_dir, "swap_consistency_table", consistency, consistency_columns),
        "swap_dataset_summary": write_table(output_dir, "swap_dataset_summary_table", dataset_summary, summary_columns),
        "hard_examples": write_table(output_dir, "hard_examples_table", examples, example_columns),
    }

    reason_counts = Counter(reason for entry in entries for reason in entry["selection_reasons"])
    report = {
        "created_at": utc_now(),
        "mode": "dry_run" if dry_run and existing_swap_scores_path is None else "scored",
        "inputs": {
            "input_dataset": path_relative_to_root(input_path),
            "calibrated_results": path_relative_to_root(calibrated_results_path),
            "base_scores": path_relative_to_root(base_scores_path),
        },
        "selection": {
            "split": "dev",
            "target_datasets": list(target_datasets),
            "low_confidence_threshold": low_confidence_threshold,
            "per_dataset_limit": per_dataset_limit,
            "selected_count": len(entries),
            "selected_by_dataset": dict(Counter(str(entry["sample"].get("dataset")) for entry in entries)),
            "selection_reason_counts": dict(reason_counts),
        },
        "scoring": scoring_summary,
        "dataset_summary": dataset_summary,
        "consistency_rows": consistency,
        "hard_examples": examples,
        "outputs": {
            "selected_samples": path_relative_to_root(selected_samples_path),
            **tables,
        },
    }
    report_path = output_dir / "swap_probe_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dev-only order-swap diagnostics for BEA-Judge.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--base-scores", type=Path, default=DEFAULT_BASE_SCORES)
    parser.add_argument("--calibrated-results", type=Path, default=None)
    parser.add_argument("--experiment-config", type=Path, default=ROOT / "configs" / "experiment.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-datasets", nargs="+", default=list(DEFAULT_TARGET_DATASETS))
    parser.add_argument("--low-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--per-dataset-limit", type=int, default=20)
    parser.add_argument("--backend", choices=["m_prometheus", "prometheus2"], default="m_prometheus")
    parser.add_argument("--model-path", type=str, default=str(ROOT / "models" / "M-Prometheus-3B"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--existing-swap-scores", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibrated_results = args.calibrated_results or resolve_latest_calibrated_results(args.experiment_config)
    report = run_probe(
        input_path=args.input,
        calibrated_results_path=calibrated_results,
        base_scores_path=args.base_scores,
        output_dir=args.output_dir,
        target_datasets=args.target_datasets,
        low_confidence_threshold=args.low_confidence_threshold,
        per_dataset_limit=args.per_dataset_limit,
        dry_run=args.dry_run,
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        existing_swap_scores_path=args.existing_swap_scores,
    )
    print(json.dumps({"selection": report["selection"], "scoring": report["scoring"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
