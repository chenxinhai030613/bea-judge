#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${PYTHON:-}" ]; then
  if [ -x "$ROOT_DIR/.venv_qlora/bin/python" ]; then
    PYTHON="$ROOT_DIR/.venv_qlora/bin/python"
  else
    PYTHON="python"
  fi
fi

SEED="${SEED:-13}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
CONFIG="${CONFIG:-configs/qlora_judge_sft_24gb_epoch1_1024.json}"
DATASET="${DATASET:-datasets/processed/bea_judge_cleaned_10000.json}"
BASE_MODEL="${BASE_MODEL:-models/M-Prometheus-3B}"
FROZEN_REPORT="${FROZEN_REPORT:-datasets/model_outputs/bea_judge_20260521_110114/validation_report.json}"
RAW_FROZEN_SUMMARY="${RAW_FROZEN_SUMMARY:-datasets/judge_outputs/m_prometheus_3b_bea10k_v2/summary.json}"
DEFAULT_ADAPTER="models/m_prometheus_3b_qlora_pairwise_seed${SEED}_epoch1${RUN_SUFFIX}"
ADAPTER="${ADAPTER:-$DEFAULT_ADAPTER}"
OUT="datasets/judge_outputs/m_prometheus_3b_qlora_pairwise_seed${SEED}_epoch1${RUN_SUFFIX}"
RUN_NAME="bea_judge_qlora_pairwise_seed${SEED}_epoch1${RUN_SUFFIX}"
COMPARISON_DIR="datasets/model_outputs/qlora_comparison_seed${SEED}_epoch1${RUN_SUFFIX}"
ARTIFACT_DIR="artifacts/qlora_seed${SEED}_epoch1${RUN_SUFFIX}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

mkdir -p logs

if [ -z "$RUN_SUFFIX" ] && [ "$SEED" = "42" ] && [ "$FORCE_TRAIN" != "1" ]; then
  LEGACY_ADAPTER="models/m_prometheus_3b_qlora_pairwise_seed42_noeval/checkpoint-301"
  if [ ! -f "$ADAPTER/adapter_model.safetensors" ] && [ -f "$LEGACY_ADAPTER/adapter_model.safetensors" ]; then
    echo "[seed $SEED] Reusing legacy epoch1 checkpoint: $LEGACY_ADAPTER"
    ADAPTER="$LEGACY_ADAPTER"
  fi
fi

echo "[seed $SEED] Checking inputs"
test -f "$CONFIG"
test -f "$DATASET"
test -d "$BASE_MODEL"
test -f "$FROZEN_REPORT"
test -f "$RAW_FROZEN_SUMMARY"

if [ "$FORCE_TRAIN" = "1" ] || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "[seed $SEED] Training 1-epoch QLoRA adapter"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  PYTHONUNBUFFERED=1 "$PYTHON" scripts/qlora_judge_train.py \
    --config "$CONFIG" \
    --output-dir "$ADAPTER" \
    --seed "$SEED" 2>&1 | tee "logs/qlora_seed${SEED}_epoch1${RUN_SUFFIX}_train.log"
else
  echo "[seed $SEED] Adapter exists; skipping train: $ADAPTER"
fi

echo "[seed $SEED] Generating train/dev/test QLoRA base scores"
mkdir -p "$OUT"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/run_qlora_base_judge.py \
  --dataset "$DATASET" \
  --split train \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output "$OUT/train_base_scores.json" 2>&1 | tee "logs/qlora_seed${SEED}_epoch1${RUN_SUFFIX}_train_scores.log"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/run_qlora_base_judge.py \
  --dataset "$DATASET" \
  --split dev \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output "$OUT/dev_base_scores.json" 2>&1 | tee "logs/qlora_seed${SEED}_epoch1${RUN_SUFFIX}_dev_scores.log"
PYTHONUNBUFFERED=1 "$PYTHON" scripts/run_qlora_base_judge.py \
  --dataset "$DATASET" \
  --split test \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER" \
  --output "$OUT/base_scores.json" 2>&1 | tee "logs/qlora_seed${SEED}_epoch1${RUN_SUFFIX}_test_scores.log"

echo "[seed $SEED] Merging base scores"
MERGE_DIR="$OUT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["MERGE_DIR"])
parts = [
    base / "train_base_scores.json",
    base / "dev_base_scores.json",
    base / "base_scores.json",
]
rows = []
seen = set()
for path in parts:
    data = json.loads(path.read_text(encoding="utf-8"))
    print(path, len(data))
    for row in data:
        rid = row.get("id")
        if rid in seen:
            raise SystemExit(f"duplicate id: {rid}")
        seen.add(rid)
        rows.append(row)
out = base / "base_scores_all.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print("merged", len(rows), "->", out)
PY

echo "[seed $SEED] Running BEA-Judge fusion/calibration"
"$PYTHON" src/bea_judge_train.py \
  --input "$DATASET" \
  --base-scores "$OUT/base_scores_all.json" \
  --run-name "$RUN_NAME"

echo "[seed $SEED] Writing comparison report"
"$PYTHON" scripts/compare_qlora_experiments.py \
  --frozen-report "$FROZEN_REPORT" \
  --qlora-report "datasets/model_outputs/${RUN_NAME}/validation_report.json" \
  --raw-frozen-summary "$RAW_FROZEN_SUMMARY" \
  --raw-qlora-scores "$OUT/base_scores.json" \
  --output-dir "$COMPARISON_DIR"

echo "[seed $SEED] Archiving key outputs"
mkdir -p "$ARTIFACT_DIR"
cp "$COMPARISON_DIR/claim_gate_report.json" "$ARTIFACT_DIR/"
cp "$COMPARISON_DIR/main_comparison_table.md" "$ARTIFACT_DIR/"
cp "$COMPARISON_DIR/qlora_comparison_report.json" "$ARTIFACT_DIR/"
cp "$OUT/summary.json" "$ARTIFACT_DIR/raw_summary.json"

echo "[seed $SEED] Complete. Inspect $COMPARISON_DIR/claim_gate_report.json"
