"""Generate BEA-compatible base_scores.json with a QLoRA judge adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from base_judge import (  # noqa: E402
    JudgeConfig,
    build_prometheus_pairwise_prompt,
    extract_prometheus_pairwise_label,
    score_from_token,
)
from dataset_adapter import samples_from_payload  # noqa: E402


PAIRWISE_LABELS = {"A>B", "B>A", "Tie"}


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_samples(path: Path, split: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    samples = samples_from_payload(json.loads(path.read_text(encoding="utf-8")))
    rows = [
        sample
        for sample in samples
        if (split is None or sample.get("split") == split)
        and sample.get("human_label") in PAIRWISE_LABELS
        and str(sample.get("answer_b") or "").strip()
    ]
    return rows[:limit] if limit is not None else rows


def label_payload(
    *,
    sample: Dict[str, Any],
    label: Optional[str],
    raw_output: str,
    parsed: Dict[str, Any],
    config: JudgeConfig,
    backend_status: Dict[str, Any],
    error: Optional[str] = None,
) -> Dict[str, Any]:
    score_a, score_b = {
        "A>B": (1.0, 0.0),
        "B>A": (0.0, 1.0),
        "Tie": (0.5, 0.5),
    }.get(label or "", (None, None))
    gold_label = sample.get("human_label")
    return {
        "id": sample.get("id"),
        "dataset": sample.get("dataset"),
        "task_type": sample.get("task_type"),
        "gold_label": gold_label,
        "pred_label": label,
        "gold_score": score_from_token(str(gold_label)) if gold_label else None,
        "pred_score": score_from_token(label or ""),
        "prompt": sample.get("prompt"),
        "context": sample.get("context"),
        "answer_a": sample.get("answer_a"),
        "answer_b": sample.get("answer_b"),
        "raw_output": raw_output,
        "judge_backend": "m_prometheus_qlora",
        "judge_name": config.name,
        "judge_version": config.version,
        "prompt_template": "m_prometheus_pairwise_v1",
        "top_p": config.top_p,
        "parsed_scores": {"score_a": score_a, "score_b": score_b},
        "parse_status": "ok" if label else "backend_error" if error else "failed",
        "backend_status": backend_status,
        "parse_metadata": parsed,
        "error": error,
    }


def repair_failed_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    repaired: List[Dict[str, Any]] = []
    score_map = {
        "A>B": (1.0, 0.0, 1.0),
        "B>A": (0.0, 1.0, -1.0),
        "Tie": (0.5, 0.5, 0.0),
    }
    for row in rows:
        if row.get("pred_label") in PAIRWISE_LABELS:
            repaired.append(dict(row))
            continue
        raw_output = str(row.get("raw_output") or "")
        label, parsed = extract_prometheus_pairwise_label(raw_output)
        if label is None:
            repaired.append(dict(row))
            continue
        score_a, score_b, pred_score = score_map[label]
        updated = dict(row)
        updated["pred_label"] = label
        updated["pred_score"] = pred_score
        updated["parsed_scores"] = {"score_a": score_a, "score_b": score_b}
        updated["parse_status"] = "retry_reparse_ok"
        updated["parse_metadata"] = parsed
        repaired.append(updated)
    return repaired


def generate_rows(
    samples: Sequence[Dict[str, Any]],
    *,
    base_model: Path,
    adapter: Path,
    max_new_tokens: int,
    retry_max_new_tokens: Optional[int],
    device_map: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing adapter inference dependencies. Install torch, transformers, peft, bitsandbytes, and accelerate."
        ) from exc

    local_files = base_model.exists()
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), use_fast=True, local_files_only=local_files)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        device_map=device_map,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=local_files,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    backend_status = {
        "available": True,
        "backend": "m_prometheus_qlora",
        "base_model": str(base_model),
        "adapter": str(adapter),
        "device_map": device_map,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "max_new_tokens": int(max_new_tokens),
        "retry_max_new_tokens": int(retry_max_new_tokens) if retry_max_new_tokens else None,
    }
    config = JudgeConfig(
        name="m_prometheus_3b_qlora_pairwise",
        version=str(adapter),
        backend="m_prometheus_qlora",
        model_path=str(base_model),
        max_new_tokens=max_new_tokens,
    )
    rows: List[Dict[str, Any]] = []

    def generate_raw(prompt: str, token_budget: int) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=token_budget,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    for sample in samples:
        prompt = build_prometheus_pairwise_prompt(sample)
        raw_output = generate_raw(prompt, max_new_tokens)
        label, parsed = extract_prometheus_pairwise_label(raw_output)
        if label is None and retry_max_new_tokens and retry_max_new_tokens > max_new_tokens:
            retry_output = generate_raw(prompt, retry_max_new_tokens)
            retry_label, retry_parsed = extract_prometheus_pairwise_label(retry_output)
            if retry_label is not None:
                raw_output = retry_output
                label = retry_label
                parsed = dict(retry_parsed)
                parsed["retry_from_max_new_tokens"] = int(max_new_tokens)
                parsed["retry_max_new_tokens"] = int(retry_max_new_tokens)
        rows.append(
            label_payload(
                sample=sample,
                label=label,
                raw_output=raw_output,
                parsed=parsed,
                config=config,
                backend_status=backend_status,
            )
        )
    return repair_failed_rows(rows), backend_status


def summarize(rows: Sequence[Dict[str, Any]], backend_status: Dict[str, Any]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("pred_label") in PAIRWISE_LABELS]
    correct = [row for row in valid if row.get("pred_label") == row.get("gold_label")]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "judge": {
            "name": "m_prometheus_3b_qlora_pairwise",
            "backend": "m_prometheus_qlora",
        },
        "backend_status": backend_status,
        "sample_count": len(rows),
        "parse_failure_count": len(rows) - len(valid),
        "overall": {
            "pairwise_accuracy": round(len(correct) / len(valid), 4) if valid else None,
            "tie_rate": round(sum(1 for row in valid if row.get("pred_label") == "Tie") / len(valid), 4)
            if valid
            else None,
        },
        "coverage": {
            "total_pairwise_samples": len(rows),
            "parsed_rows": len(valid),
            "failed_rows": len(rows) - len(valid),
            "parse_success_rate": round(len(valid) / len(rows), 4) if rows else 0.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a QLoRA adapter as a BEA base judge.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default=None, choices=["train", "dev", "test"])
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--retry-max-new-tokens", type=int, default=96)
    parser.add_argument("--device-map", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(resolve_root_path(args.dataset), args.split, args.limit)
    rows, backend_status = generate_rows(
        samples,
        base_model=resolve_root_path(args.base_model),
        adapter=resolve_root_path(args.adapter),
        max_new_tokens=args.max_new_tokens,
        retry_max_new_tokens=args.retry_max_new_tokens,
        device_map=args.device_map,
    )
    output = resolve_root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(rows, backend_status)
    (output.parent / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
