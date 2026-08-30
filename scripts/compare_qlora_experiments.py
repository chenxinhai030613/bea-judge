"""Compare frozen BEA-Judge and QLoRA-BEA-Judge experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bea_judge_train import PAIRWISE_LABELS, ece_score_from_predictions, macro_f1  # noqa: E402


CURRENT_PAIRWISE_BASELINE = {
    "accuracy": 0.7512,
    "macro_f1": 0.673,
    "ece": 0.0558,
    "tie_recall": 0.5231,
}
RAW_FROZEN_FALLBACK = {
    "accuracy": 0.5632,
    "macro_f1": 0.4079,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def metric_from_report(report: Dict[str, Any], head: str = "pairwise", split: str = "test") -> Dict[str, Any]:
    if split == "test":
        return dict(report.get("test_evaluation", {}).get(head, {}).get("metrics", {}))
    if split == "dev":
        return dict(report.get("heads", {}).get(head, {}).get("calibrated_dev_metrics", {}))
    raise ValueError(f"unsupported split: {split}")


def raw_pairwise_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    label_to_idx = {label: i for i, label in enumerate(PAIRWISE_LABELS)}
    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []
    for row in rows:
        gold = row.get("gold_label")
        pred = row.get("pred_label")
        if gold not in label_to_idx or pred not in label_to_idx:
            continue
        y_true.append(label_to_idx[str(gold)])
        y_pred.append(label_to_idx[str(pred)])
        scores = row.get("parsed_scores", {}) or {}
        score_a = float(scores.get("score_a") or 0.0)
        score_b = float(scores.get("score_b") or 0.0)
        confidences.append(max(score_a, score_b, 0.5 if pred == "Tie" else 0.0))
    if not y_true:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
            "ece": None,
            "tie_recall": None,
            "parse_failure_rate": 1.0,
        }
    import numpy as np

    y_true_array = np.array(y_true, dtype=int)
    y_pred_array = np.array(y_pred, dtype=int)
    conf_array = np.array(confidences, dtype=float)
    tie_index = label_to_idx["Tie"]
    tie_mask = y_true_array == tie_index
    return {
        "n": len(y_true),
        "accuracy": round(float((y_true_array == y_pred_array).mean()), 6),
        "macro_f1": round(float(macro_f1(y_true_array, y_pred_array, list(range(len(PAIRWISE_LABELS))))), 6),
        "ece": round(float(ece_score_from_predictions(y_true_array, y_pred_array, conf_array)), 6),
        "tie_recall": round(float((y_pred_array[tie_mask] == tie_index).mean()), 6) if tie_mask.any() else None,
        "parse_failure_rate": round(
            sum(1 for row in rows if row.get("pred_label") not in label_to_idx) / len(rows),
            6,
        )
        if rows
        else 1.0,
        "pred_label_distribution": dict(Counter(row.get("pred_label") for row in rows)),
    }


def markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def build_comparison(
    *,
    frozen_report: Dict[str, Any],
    qlora_report: Dict[str, Any],
    raw_frozen_summary: Optional[Dict[str, Any]],
    raw_qlora_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    frozen_pairwise = metric_from_report(frozen_report, "pairwise", "test") or CURRENT_PAIRWISE_BASELINE
    qlora_pairwise = metric_from_report(qlora_report, "pairwise", "test")
    raw_qlora = raw_pairwise_metrics(raw_qlora_rows)
    raw_qlora_parse_failure_rate = raw_qlora.get("parse_failure_rate")
    if raw_qlora_parse_failure_rate is None:
        raw_qlora_parse_failure_rate = 1.0
    raw_frozen = dict(RAW_FROZEN_FALLBACK)
    if raw_frozen_summary:
        overall = raw_frozen_summary.get("overall", {})
        if overall.get("pairwise_accuracy") is not None:
            raw_frozen["accuracy"] = overall.get("pairwise_accuracy")

    rows = [
        {
            "system": "Raw M-Prometheus-3B",
            "base": "frozen",
            "four_module": "no",
            "accuracy": raw_frozen.get("accuracy"),
            "macro_f1": raw_frozen.get("macro_f1"),
            "ece": raw_frozen.get("ece", ""),
            "tie_recall": raw_frozen.get("tie_recall", ""),
        },
        {
            "system": "Current BEA-Judge",
            "base": "frozen",
            "four_module": "yes",
            "accuracy": frozen_pairwise.get("accuracy"),
            "macro_f1": frozen_pairwise.get("macro_f1"),
            "ece": frozen_pairwise.get("ece"),
            "tie_recall": frozen_pairwise.get("tie_recall"),
        },
        {
            "system": "QLoRA-M-Prometheus-3B",
            "base": "qlora",
            "four_module": "no",
            "accuracy": raw_qlora.get("accuracy"),
            "macro_f1": raw_qlora.get("macro_f1"),
            "ece": raw_qlora.get("ece"),
            "tie_recall": raw_qlora.get("tie_recall"),
        },
        {
            "system": "QLoRA-BEA-Judge",
            "base": "qlora",
            "four_module": "yes",
            "accuracy": qlora_pairwise.get("accuracy"),
            "macro_f1": qlora_pairwise.get("macro_f1"),
            "ece": qlora_pairwise.get("ece"),
            "tie_recall": qlora_pairwise.get("tie_recall"),
        },
    ]
    checks = {
        "qlora_raw_macro_f1_gain_min_0_10": (
            raw_qlora.get("macro_f1") is not None
            and raw_frozen.get("macro_f1") is not None
            and float(raw_qlora["macro_f1"]) - float(raw_frozen["macro_f1"]) >= 0.10
        ),
        "qlora_bea_macro_f1_gain_min_0_02": (
            qlora_pairwise.get("macro_f1") is not None
            and frozen_pairwise.get("macro_f1") is not None
            and float(qlora_pairwise["macro_f1"]) - float(frozen_pairwise["macro_f1"]) >= 0.02
        ),
        "tie_recall_not_below_baseline": (
            qlora_pairwise.get("tie_recall") is not None
            and float(qlora_pairwise["tie_recall"]) >= CURRENT_PAIRWISE_BASELINE["tie_recall"]
        ),
        "ece_max_0_06": (
            qlora_pairwise.get("ece") is not None and float(qlora_pairwise["ece"]) <= 0.06
        ),
        "raw_qlora_parse_failure_max_0_01": float(raw_qlora_parse_failure_rate) <= 0.01,
    }
    return {
        "comparison_rows": rows,
        "raw_qlora_metrics": raw_qlora,
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
        },
    }


def write_outputs(result: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "main_comparison_table.md").write_text(
        markdown_table(result["comparison_rows"]),
        encoding="utf-8",
    )
    (output_dir / "claim_gate_report.json").write_text(
        json.dumps(result["gate"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "qlora_comparison_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QLoRA and frozen BEA-Judge reports.")
    parser.add_argument("--frozen-report", type=str, required=True)
    parser.add_argument("--qlora-report", type=str, required=True)
    parser.add_argument("--raw-frozen-summary", type=str, default=None)
    parser.add_argument("--raw-qlora-scores", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frozen_report = load_json(resolve_root_path(args.frozen_report))
    qlora_report = load_json(resolve_root_path(args.qlora_report))
    raw_frozen_summary = load_json(resolve_root_path(args.raw_frozen_summary)) if args.raw_frozen_summary else None
    raw_qlora_rows = load_json(resolve_root_path(args.raw_qlora_scores))
    result = build_comparison(
        frozen_report=frozen_report,
        qlora_report=qlora_report,
        raw_frozen_summary=raw_frozen_summary,
        raw_qlora_rows=raw_qlora_rows,
    )
    write_outputs(result, resolve_root_path(args.output_dir))
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
