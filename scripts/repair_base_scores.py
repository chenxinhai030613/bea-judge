from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from base_judge import (  # noqa: E402
    JudgeConfig,
    extract_prometheus_pairwise_label,
    is_valid_pairwise_score_row,
    pairwise_samples,
    read_dataset,
    run_resumable_evaluation,
    split_base_score_rows,
)


DEFAULT_INPUT = ROOT / "datasets" / "processed" / "bea_judge_cleaned_3400.json"
DEFAULT_SOURCE_RUN = ROOT / "datasets" / "judge_outputs" / "m_prometheus_3b_real_full_promptfix_256"


def resolve_workspace_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    resolved = resolve_workspace_path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"expected all rows to be objects: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def rows_by_id(rows: Sequence[Dict[str, Any]], *, valid_only: bool) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("id")
        if sample_id is None:
            continue
        if valid_only and not is_valid_pairwise_score_row(row):
            continue
        out[str(sample_id)] = row
    return out


def failed_ids(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(row["id"])
        for row in rows
        if row.get("id") is not None and not is_valid_pairwise_score_row(row)
    ]


def select_samples_by_id(samples: Sequence[Dict[str, Any]], ids: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = {str(sample_id) for sample_id in ids}
    return [sample for sample in samples if str(sample.get("id")) in wanted]


def mark_retry_rows(rows: Sequence[Dict[str, Any]], max_new_tokens: int) -> List[Dict[str, Any]]:
    marked: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["parse_status"] = "retry_ok"
        copied["repair_metadata"] = {
            "source": "m_prometheus_retry",
            "max_new_tokens": int(max_new_tokens),
        }
        marked.append(copied)
    return marked


def reparse_retry_rows(rows: Sequence[Dict[str, Any]], max_new_tokens: int) -> List[Dict[str, Any]]:
    reparsed: List[Dict[str, Any]] = []
    score_map = {
        "A>B": (1.0, 0.0),
        "B>A": (0.0, 1.0),
        "Tie": (0.5, 0.5),
    }
    for row in rows:
        if is_valid_pairwise_score_row(row):
            reparsed.append(row)
            continue
        raw_output = str(row.get("raw_output") or "")
        label, parsed = extract_prometheus_pairwise_label(raw_output)
        if label is None:
            reparsed.append(row)
            continue
        score_a, score_b = score_map[label]
        copied = dict(row)
        copied["pred_label"] = label
        copied["pred_score"] = score_a
        copied["parsed_scores"] = {"score_a": score_a, "score_b": score_b}
        copied["parse_status"] = "retry_reparse_ok"
        copied["parse_metadata"] = parsed
        copied["repair_metadata"] = {
            "source": "m_prometheus_retry_reparse",
            "max_new_tokens": int(max_new_tokens),
        }
        reparsed.append(copied)
    return reparsed


def merge_repaired_rows(
    required_ids: Sequence[str],
    original_rows: Sequence[Dict[str, Any]],
    retry_rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    original_valid = rows_by_id(original_rows, valid_only=True)
    retry_valid = rows_by_id(retry_rows, valid_only=True)
    repaired: List[Dict[str, Any]] = []
    replaced: List[str] = []
    unresolved: List[str] = []

    for sample_id in required_ids:
        if sample_id in retry_valid:
            repaired.append(retry_valid[sample_id])
            replaced.append(sample_id)
        elif sample_id in original_valid:
            repaired.append(original_valid[sample_id])
        else:
            unresolved.append(sample_id)

    return repaired, replaced, unresolved


def coverage_report(required_ids: Sequence[str], rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = rows_by_id(rows, valid_only=True)
    missing = [sample_id for sample_id in required_ids if sample_id not in valid]
    return {
        "required_pairwise_rows": len(required_ids),
        "covered_pairwise_rows": len(required_ids) - len(missing),
        "missing_pairwise_rows": len(missing),
        "missing_examples": missing[:10],
        "coverage_passed": not missing,
    }


def retry_failed_samples(
    samples: Sequence[Dict[str, Any]],
    *,
    run_dir: Path,
    model_path: str,
    max_new_tokens: int,
    checkpoint_interval: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = JudgeConfig(
        name=f"m_prometheus_3b_repair_{max_new_tokens}",
        backend="m_prometheus",
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        allow_fallback=False,
    )
    summary, rows = run_resumable_evaluation(
        samples,
        config,
        run_dir,
        checkpoint_interval=checkpoint_interval,
    )
    reparsed_rows = reparse_retry_rows(rows, max_new_tokens)
    valid_rows, _failures = split_base_score_rows(reparsed_rows)
    return mark_retry_rows(valid_rows, max_new_tokens), summary


def repair_base_scores(
    *,
    input_path: Path,
    source_run_dir: Path,
    output_scores_path: Path,
    output_report_path: Path,
    model_path: str,
    retry_tokens: Sequence[int],
    checkpoint_interval: int,
) -> Dict[str, Any]:
    input_path = resolve_workspace_path(input_path)
    source_run_dir = resolve_workspace_path(source_run_dir)
    output_scores_path = resolve_workspace_path(output_scores_path)
    output_report_path = resolve_workspace_path(output_report_path)
    payload = read_dataset(input_path)
    required_samples = pairwise_samples(payload["samples"])
    required_ids = [str(sample["id"]) for sample in required_samples]
    original_scores_path = source_run_dir / "base_scores.json"
    original_rows = read_json_list(original_scores_path)
    pending_ids = failed_ids(original_rows)

    retry_rows: List[Dict[str, Any]] = []
    retry_reports: List[Dict[str, Any]] = []
    for max_new_tokens in retry_tokens:
        if not pending_ids:
            break
        retry_samples = select_samples_by_id(required_samples, pending_ids)
        stage_dir = source_run_dir / f"repair_retry_{max_new_tokens}"
        stage_rows, stage_summary = retry_failed_samples(
            retry_samples,
            run_dir=stage_dir,
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            checkpoint_interval=checkpoint_interval,
        )
        retry_rows.extend(stage_rows)
        merged_for_stage, _replaced, unresolved = merge_repaired_rows(
            required_ids,
            original_rows,
            retry_rows,
        )
        pending_ids = unresolved
        retry_reports.append(
            {
                "max_new_tokens": max_new_tokens,
                "run_dir": display_path(stage_dir),
                "attempted_rows": len(retry_samples),
                "valid_retry_rows": len(stage_rows),
                "remaining_unresolved_rows": len(pending_ids),
                "summary": stage_summary,
                "intermediate_coverage": coverage_report(required_ids, merged_for_stage),
            }
        )

    repaired_rows, replaced_ids, unresolved_ids = merge_repaired_rows(required_ids, original_rows, retry_rows)
    report = {
        "input_dataset": display_path(input_path),
        "source_base_scores": display_path(original_scores_path),
        "output_base_scores": display_path(output_scores_path),
        "initial_failed_rows": len(failed_ids(original_rows)),
        "retry_tokens": list(retry_tokens),
        "retry_reports": retry_reports,
        "replaced_rows": len(replaced_ids),
        "replaced_examples": replaced_ids[:10],
        "unresolved_rows": len(unresolved_ids),
        "unresolved_examples": unresolved_ids[:20],
        "coverage": coverage_report(required_ids, repaired_rows),
    }
    write_json(output_report_path, report)
    if unresolved_ids:
        raise RuntimeError(f"base score repair unresolved {len(unresolved_ids)} rows; examples={unresolved_ids[:10]}")
    write_json(output_scores_path, repaired_rows)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry failed M-Prometheus rows and merge repaired base scores.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--model-path", type=str, default="models\\M-Prometheus-3B")
    parser.add_argument("--retry-tokens", type=int, nargs="+", default=[512, 768])
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run_dir = args.source_run_dir
    output = args.output or source_run_dir / "base_scores.repaired.json"
    report = args.report or source_run_dir / "base_scores_repair_report.json"
    result = repair_base_scores(
        input_path=args.input,
        source_run_dir=source_run_dir,
        output_scores_path=output,
        output_report_path=report,
        model_path=args.model_path,
        retry_tokens=args.retry_tokens,
        checkpoint_interval=args.checkpoint_interval,
    )
    print(json.dumps(result["coverage"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
