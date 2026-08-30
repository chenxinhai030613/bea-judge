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
EXPERIMENT_TAG="${EXPERIMENT_TAG:?EXPERIMENT_TAG is required}"
CONFIG="${CONFIG:-configs/qlora_judge_sft_24gb_epoch1_1024.json}"
DATASET="${DATASET:-datasets/processed/bea_judge_cleaned_10000.json}"
BASE_MODEL="${BASE_MODEL:-models/M-Prometheus-3B}"
FROZEN_REPORT="${FROZEN_REPORT:-datasets/model_outputs/bea_judge_20260521_110114/validation_report.json}"
RAW_FROZEN_SUMMARY="${RAW_FROZEN_SUMMARY:-datasets/judge_outputs/m_prometheus_3b_bea10k_v2/summary.json}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-datasets/sft/m_prometheus_pairwise}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_SCORE_SAMPLES="${MAX_SCORE_SAMPLES:-}"
MAX_FUSION_SAMPLES="${MAX_FUSION_SAMPLES:-${MAX_SCORE_SAMPLES:-}}"
ADAPTER="${ADAPTER:-models/m_prometheus_3b_qlora_pairwise_seed${SEED}_${EXPERIMENT_TAG}}"
OUT="${OUT:-datasets/judge_outputs/m_prometheus_3b_qlora_pairwise_seed${SEED}_${EXPERIMENT_TAG}}"
RUN_NAME="${RUN_NAME:-bea_judge_qlora_pairwise_seed${SEED}_${EXPERIMENT_TAG}}"
COMPARISON_DIR="${COMPARISON_DIR:-datasets/model_outputs/qlora_comparison_seed${SEED}_${EXPERIMENT_TAG}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/qlora_seed${SEED}_${EXPERIMENT_TAG}}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"

mkdir -p logs

echo "[ablation seed $SEED] Checking inputs"
test -f "$CONFIG"
test -f "$DATASET"
test -d "$BASE_MODEL"
test -f "$FROZEN_REPORT"
test -f "$RAW_FROZEN_SUMMARY"
test -f "$SFT_OUTPUT_DIR/train.jsonl"
test -f "$SFT_OUTPUT_DIR/dev.jsonl"

TRAIN_ARGS=()
if [ -n "$NUM_TRAIN_EPOCHS" ]; then
  TRAIN_ARGS+=(--num-train-epochs "$NUM_TRAIN_EPOCHS")
fi
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
  TRAIN_ARGS+=(--max-samples "$MAX_TRAIN_SAMPLES")
fi

SCORE_ARGS=()
if [ -n "$MAX_SCORE_SAMPLES" ]; then
  SCORE_ARGS+=(--limit "$MAX_SCORE_SAMPLES")
fi

FUSION_ARGS=()
if [ -n "$MAX_FUSION_SAMPLES" ]; then
  FUSION_ARGS+=(--sample-limit "$MAX_FUSION_SAMPLES")
fi

if [ "$FORCE_TRAIN" = "1" ] || [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "[ablation seed $SEED] Training QLoRA adapter for $EXPERIMENT_TAG"
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  PYTHONUNBUFFERED=1 "$PYTHON" scripts/qlora_judge_train.py \
    --config "$CONFIG" \
    --output-dir "$ADAPTER" \
    --seed "$SEED" \
    --sft-output-dir "$SFT_OUTPUT_DIR" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "logs/qlora_seed${SEED}_${EXPERIMENT_TAG}_train.log"
else
  echo "[ablation seed $SEED] Adapter exists; skipping train: $ADAPTER"
fi

echo "[ablation seed $SEED] Generating train/dev/test QLoRA base scores"
mkdir -p "$OUT"
for split in train dev test; do
  split_out="$OUT/base_scores.json"
  split_log="logs/qlora_seed${SEED}_${EXPERIMENT_TAG}_test_scores.log"
  if [ "$split" = "train" ]; then
    split_out="$OUT/train_base_scores.json"
    split_log="logs/qlora_seed${SEED}_${EXPERIMENT_TAG}_train_scores.log"
  elif [ "$split" = "dev" ]; then
    split_out="$OUT/dev_base_scores.json"
    split_log="logs/qlora_seed${SEED}_${EXPERIMENT_TAG}_dev_scores.log"
  fi
  PYTHONUNBUFFERED=1 "$PYTHON" scripts/run_qlora_base_judge.py \
    --dataset "$DATASET" \
    --split "$split" \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --output "$split_out" \
    "${SCORE_ARGS[@]}" 2>&1 | tee "$split_log"
done

echo "[ablation seed $SEED] Merging base scores"
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

echo "[ablation seed $SEED] Running BEA-Judge fusion/calibration"
"$PYTHON" src/bea_judge_train.py \
  --input "$DATASET" \
  --base-scores "$OUT/base_scores_all.json" \
  --run-name "$RUN_NAME" \
  "${FUSION_ARGS[@]}"

echo "[ablation seed $SEED] Writing comparison report"
"$PYTHON" scripts/compare_qlora_experiments.py \
  --frozen-report "$FROZEN_REPORT" \
  --qlora-report "datasets/model_outputs/${RUN_NAME}/validation_report.json" \
  --raw-frozen-summary "$RAW_FROZEN_SUMMARY" \
  --raw-qlora-scores "$OUT/base_scores.json" \
  --output-dir "$COMPARISON_DIR"

echo "[ablation seed $SEED] Archiving key outputs"
mkdir -p "$ARTIFACT_DIR"
cp "$COMPARISON_DIR/claim_gate_report.json" "$ARTIFACT_DIR/"
cp "$COMPARISON_DIR/main_comparison_table.md" "$ARTIFACT_DIR/"
cp "$COMPARISON_DIR/qlora_comparison_report.json" "$ARTIFACT_DIR/"
cp "$OUT/summary.json" "$ARTIFACT_DIR/raw_summary.json"

echo "[ablation seed $SEED] Complete. Inspect $COMPARISON_DIR/claim_gate_report.json"
