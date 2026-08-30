"""Train a QLoRA pairwise judge adapter for M-Prometheus-3B."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent
from path_utils import resolve_project_path
ALLOWED_TARGETS = {"[RESULT] A", "[RESULT] B", "[RESULT] Tie"}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_root_path(value: str) -> Path:
    return resolve_project_path(ROOT, value)


def load_jsonl(path: Path, *, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target_text") not in ALLOWED_TARGETS:
                raise ValueError(f"unsupported target_text in {path}: {row.get('target_text')!r}")
            rows.append(row)
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def encode_sft_example(
    *,
    prompt_text: str,
    target_text: str,
    tokenizer: Any,
    max_length: int,
) -> Dict[str, List[int]]:
    prompt = f"{prompt_text.rstrip()}\n"
    target = f"{target_text}{tokenizer.eos_token}"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    prompt_budget = max(1, max_length - len(target_ids))
    prompt_ids = prompt_ids[-prompt_budget:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def require_training_dependencies() -> None:
    missing = []
    for package in ("torch", "transformers", "datasets", "peft", "bitsandbytes"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "Missing QLoRA training dependencies: "
            + ", ".join(missing)
            + ". Install on the CUDA server with: pip install torch transformers accelerate datasets peft bitsandbytes trl safetensors"
        )


def train(
    config: Dict[str, Any],
    *,
    output_dir: Optional[str],
    seed: Optional[int],
    max_samples: Optional[int],
    sft_output_dir: Optional[str],
) -> Path:
    require_training_dependencies()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    data_config = config["data"]
    qlora = config["qlora"]
    training = config["training"]
    resolved_seed = int(seed if seed is not None else training.get("seed", 42))
    set_seed(resolved_seed)
    random.seed(resolved_seed)

    base_model = str(resolve_root_path(config["base_model"]))
    run_dir = resolve_root_path(output_dir or config["output_dir"])
    sft_dir = resolve_root_path(sft_output_dir or data_config["output_dir"])
    train_rows = load_jsonl(sft_dir / "train.jsonl", max_samples=max_samples)
    dev_rows = load_jsonl(sft_dir / "dev.jsonl", max_samples=max_samples)
    if not train_rows:
        raise ValueError("no SFT train rows found; run scripts/build_judge_sft_dataset.py first")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, local_files_only=Path(base_model).exists())
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if qlora.get("bnb_4bit_compute_dtype") == "bfloat16" else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=bool(qlora.get("load_in_4bit", True)),
        bnb_4bit_quant_type=str(qlora.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(qlora.get("bnb_4bit_use_double_quant", True)),
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=compute_dtype,
        local_files_only=Path(base_model).exists(),
    )
    model.config.use_cache = False
    if bool(training.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=int(qlora.get("lora_r", 16)),
        lora_alpha=int(qlora.get("lora_alpha", 32)),
        lora_dropout=float(qlora.get("lora_dropout", 0.05)),
        target_modules=list(qlora.get("target_modules", ["q_proj", "v_proj", "o_proj"])),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    max_length = int(training.get("max_seq_length", 2048))

    def encode_row(row: Dict[str, Any]) -> Dict[str, List[int]]:
        return encode_sft_example(
            prompt_text=row["prompt_text"],
            target_text=row["target_text"],
            tokenizer=tokenizer,
            max_length=max_length,
        )

    def collate(batch: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        max_batch_len = max(len(row["input_ids"]) for row in batch)
        input_ids = []
        attention_mask = []
        labels = []
        for row in batch:
            pad = max_batch_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [tokenizer.pad_token_id] * pad)
            attention_mask.append(row["attention_mask"] + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_dataset = Dataset.from_list([encode_row(row) for row in train_rows])
    eval_dataset = Dataset.from_list([encode_row(row) for row in dev_rows])

    eval_strategy_key = "eval_strategy"
    try:
        TrainingArguments(output_dir=str(run_dir), eval_strategy="no")
    except TypeError:
        eval_strategy_key = "evaluation_strategy"

    args_payload = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 16)),
        "learning_rate": float(training.get("learning_rate", 1e-4)),
        "num_train_epochs": float(training.get("num_train_epochs", 2)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "lr_scheduler_type": str(training.get("lr_scheduler_type", "cosine")),
        "logging_steps": int(training.get("logging_steps", 20)),
        "save_strategy": str(training.get("save_strategy", "epoch")),
        "save_total_limit": int(training.get("save_total_limit", 2)),
        "bf16": compute_dtype is torch.bfloat16,
        "fp16": compute_dtype is torch.float16,
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "report_to": [],
        "seed": resolved_seed,
    }
    args_payload[eval_strategy_key] = "steps"
    args_payload["eval_steps"] = int(training.get("eval_steps", 200))
    train_args = TrainingArguments(**args_payload)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
    )
    trainer.train()
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))
    metadata = {
        "base_model": base_model,
        "output_dir": str(run_dir),
        "sft_output_dir": str(sft_dir),
        "seed": resolved_seed,
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "max_seq_length": max_length,
        "qlora": qlora,
        "training": training,
    }
    (run_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QLoRA adapter for pairwise judge SFT.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "qlora_judge_sft.json"))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sft-output-dir", type=str, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None, help="Override config for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(resolve_root_path(args.config))
    if args.num_train_epochs is not None:
        config.setdefault("training", {})["num_train_epochs"] = args.num_train_epochs
    run_dir = train(
        config,
        output_dir=args.output_dir,
        seed=args.seed,
        max_samples=args.max_samples,
        sft_output_dir=args.sft_output_dir,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
