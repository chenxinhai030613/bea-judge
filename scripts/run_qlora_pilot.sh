#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/qlora_judge_sft.json}"
DATASET="${DATASET:-datasets/processed/bea_judge_cleaned_10000.json}"
BASE_MODEL="${BASE_MODEL:-models/M-Prometheus-3B}"
SEED="${SEED:-42}"
SMOKE_ADAPTER="${SMOKE_ADAPTER:-models/m_prometheus_3b_qlora_smoke_seed42}"
ADAPTER="${ADAPTER:-models/m_prometheus_3b_qlora_pairwise_seed42}"
SCORES_DIR="${SCORES_DIR:-datasets/judge_outputs/m_prometheus_3b_qlora_pairwise_seed42}"
RUN_NAME="${RUN_NAME:-bea_judge_qlora_pairwise_seed42}"
COMPARISON_DIR="${COMPARISON_DIR:-datasets/model_outputs/qlora_comparison_seed42}"

echo "[1/7] Checking CUDA and QLoRA dependencies"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "transformers", "accelerate", "datasets", "peft", "bitsandbytes") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing dependencies: {', '.join(missing)}")
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}; VRAM_GB={props.total_memory / 1024**3:.2f}")
PY

echo "[2/7] Exporting pairwise SFT data"
"$PYTHON_BIN" scripts/build_judge_sft_dataset.py --config "$CONFIG"

echo "[3/7] Running 32-sample QLoRA smoke train"
"$PYTHON_BIN" scripts/qlora_judge_train.py \
  --config "$CONFIG" \
  --output-dir "$SMOKE_ADAPTER" \
  --seed "$SEED" \
  --max-samples 32 \
  --num-train-epochs 0.01

echo "[4/7] Running seed${SEED} QLoRA pilot train"
"$PYTHON_BIN" scripts/qlora_judge_train.py \
  --config "$CONFIG" \
  --output-dir "$ADAPTER" \
  --seed "$SEED"

echo "[5/7] Generating QLoRA test base scores"
"$PYTHON_BIN" scripts/run_qlora_base_judge.py \
  --dataset "$DATASET" \
  --split test \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output "$SCORES_DIR/base_scores.json"

echo "[6/7] Training BEA-Judge fusion/calibration with QLoRA base scores"
"$PYTHON_BIN" src/bea_judge_train.py \
  --input "$DATASET" \
  --base-scores "$SCORES_DIR/base_scores.json" \
  --run-name "$RUN_NAME"

echo "[7/7] Comparing QLoRA pilot against frozen baseline"
"$PYTHON_BIN" scripts/compare_qlora_experiments.py \
  --frozen-report datasets/model_outputs/bea_judge_20260521_110114/validation_report.json \
  --qlora-report "datasets/model_outputs/${RUN_NAME}/validation_report.json" \
  --raw-frozen-summary datasets/judge_outputs/m_prometheus_3b_bea10k_v2/summary.json \
  --raw-qlora-scores "$SCORES_DIR/base_scores.json" \
  --output-dir "$COMPARISON_DIR"

echo "Pilot complete. Inspect $COMPARISON_DIR/claim_gate_report.json"
