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

GROUP="${GROUP:-all}"
SEEDS="${SEEDS:-13 42 2026}"
CONFIG="${CONFIG:-configs/qlora_judge_sft_24gb_epoch1_1024.json}"
FORCE_RUN="${FORCE_RUN:-0}"
FORCE_ALIAS="${FORCE_ALIAS:-0}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
SFT_SOURCE_DIR="${SFT_SOURCE_DIR:-datasets/sft/m_prometheus_pairwise}"
SFT_OUTPUT_ROOT="${SFT_OUTPUT_ROOT:-datasets/sft}"
SFT_SUBSET_SIZES="${SFT_SUBSET_SIZES:-1202 2403}"
SFT_SUBSET_SEED="${SFT_SUBSET_SEED:-42}"

case "$GROUP" in
  all|epoch|sft_size)
    ;;
  *)
    echo "Unsupported GROUP=$GROUP (expected: all, epoch, sft_size)" >&2
    exit 1
    ;;
esac

comparison_report_path() {
  local seed="$1"
  local setting="$2"
  echo "datasets/model_outputs/qlora_comparison_seed${seed}_${setting}/qlora_comparison_report.json"
}

require_reference_report() {
  local seed="$1"
  local setting="$2"
  local report
  report="$(comparison_report_path "$seed" "$setting")"
  if [ ! -f "$report" ]; then
    echo "Missing reference report: $report" >&2
    exit 1
  fi
}

run_setting() {
  local seed="$1"
  local setting="$2"
  local sft_dir="$3"
  local num_train_epochs="${4:-}"
  local report
  report="$(comparison_report_path "$seed" "$setting")"
  if [ "$FORCE_RUN" != "1" ] && [ -f "$report" ]; then
    echo "[grid] Reusing existing report for seed=$seed setting=$setting"
    return
  fi

  echo "[grid] Running seed=$seed setting=$setting sft_dir=$sft_dir num_train_epochs=${num_train_epochs:-config}"
  if [ -n "$num_train_epochs" ]; then
    SEED="$seed" \
    EXPERIMENT_TAG="$setting" \
    CONFIG="$CONFIG" \
    SFT_OUTPUT_DIR="$sft_dir" \
    NUM_TRAIN_EPOCHS="$num_train_epochs" \
    bash scripts/run_qlora_ablation_seed.sh
  else
    SEED="$seed" \
    EXPERIMENT_TAG="$setting" \
    CONFIG="$CONFIG" \
    SFT_OUTPUT_DIR="$sft_dir" \
    bash scripts/run_qlora_ablation_seed.sh
  fi
}

alias_setting() {
  local seed="$1"
  local src_setting="$2"
  local dst_setting="$3"
  local src_dir="datasets/model_outputs/qlora_comparison_seed${seed}_${src_setting}"
  local dst_dir="datasets/model_outputs/qlora_comparison_seed${seed}_${dst_setting}"

  require_reference_report "$seed" "$src_setting"

  if [ "$FORCE_ALIAS" != "1" ] && [ -f "$dst_dir/qlora_comparison_report.json" ]; then
    echo "[grid] Reusing existing alias for seed=$seed setting=$dst_setting"
    return
  fi

  echo "[grid] Aliasing seed=$seed $dst_setting -> $src_setting"
  mkdir -p "$dst_dir"
  cp "$src_dir/qlora_comparison_report.json" "$dst_dir/qlora_comparison_report.json"
  if [ -f "$src_dir/claim_gate_report.json" ]; then
    cp "$src_dir/claim_gate_report.json" "$dst_dir/claim_gate_report.json"
  fi
  if [ -f "$src_dir/main_comparison_table.md" ]; then
    cp "$src_dir/main_comparison_table.md" "$dst_dir/main_comparison_table.md"
  fi
  cat > "$dst_dir/alias_metadata.json" <<EOF
{
  "seed": "$seed",
  "setting": "$dst_setting",
  "alias_from_setting": "$src_setting",
  "config": "$CONFIG",
  "note": "Reused the stable epoch1_1024 three-seed result as the 100% SFT-size reference without retraining."
}
EOF
}

ensure_sft_subsets() {
  if [ -f "$SFT_OUTPUT_ROOT/m_prometheus_pairwise_sft25/train.jsonl" ] && [ -f "$SFT_OUTPUT_ROOT/m_prometheus_pairwise_sft50/train.jsonl" ]; then
    echo "[grid] Reusing existing SFT subsets under $SFT_OUTPUT_ROOT"
    return
  fi

  echo "[grid] Building deterministic SFT subsets"
  "$PYTHON" scripts/build_qlora_sft_subsets.py \
    --source-dir "$SFT_SOURCE_DIR" \
    --output-root "$SFT_OUTPUT_ROOT" \
    --sample-sizes $SFT_SUBSET_SIZES \
    --seed "$SFT_SUBSET_SEED"
}

run_epoch_group() {
  echo "[grid] Starting epoch ablation for seeds: $SEEDS"
  for seed in $SEEDS; do
    require_reference_report "$seed" "epoch1_1024"
    run_setting "$seed" "epoch0p5_1024" "$SFT_SOURCE_DIR" "0.5"
    run_setting "$seed" "epoch2_1024" "$SFT_SOURCE_DIR" "2"
  done

  if [ "$RUN_POSTPROCESS" = "1" ]; then
    echo "[grid] Summarizing epoch ablation"
    "$PYTHON" scripts/summarize_qlora_ablation_grid.py \
      --settings epoch0p5_1024 epoch1_1024 epoch2_1024 \
      --seeds $SEEDS \
      --report-template "datasets/model_outputs/qlora_comparison_seed{seed}_{setting}/qlora_comparison_report.json" \
      --output-json "datasets/model_outputs/qlora_epoch_ablation_3seed_1024_summary/epoch_ablation_summary.json" \
      --output-md "datasets/model_outputs/qlora_epoch_ablation_3seed_1024_summary/epoch_ablation_summary.md" \
      --title "QLoRA Epoch Ablation Summary"
  fi
}

run_sft_size_group() {
  echo "[grid] Starting SFT-size ablation for seeds: $SEEDS"
  ensure_sft_subsets
  for seed in $SEEDS; do
    require_reference_report "$seed" "epoch1_1024"
    run_setting "$seed" "sft25_epoch1_1024" "$SFT_OUTPUT_ROOT/m_prometheus_pairwise_sft25"
    run_setting "$seed" "sft50_epoch1_1024" "$SFT_OUTPUT_ROOT/m_prometheus_pairwise_sft50"
    alias_setting "$seed" "epoch1_1024" "sft100_epoch1_1024"
  done

  if [ "$RUN_POSTPROCESS" = "1" ]; then
    echo "[grid] Summarizing SFT-size ablation"
    "$PYTHON" scripts/summarize_qlora_ablation_grid.py \
      --settings sft25_epoch1_1024 sft50_epoch1_1024 sft100_epoch1_1024 \
      --seeds $SEEDS \
      --report-template "datasets/model_outputs/qlora_comparison_seed{seed}_{setting}/qlora_comparison_report.json" \
      --output-json "datasets/model_outputs/qlora_sft_size_ablation_3seed_1024_summary/sft_size_ablation_summary.json" \
      --output-md "datasets/model_outputs/qlora_sft_size_ablation_3seed_1024_summary/sft_size_ablation_summary.md" \
      --title "QLoRA SFT Size Ablation Summary"
  fi
}

if [ "$GROUP" = "all" ] || [ "$GROUP" = "epoch" ]; then
  run_epoch_group
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "sft_size" ]; then
  run_sft_size_group
fi

echo "[grid] Complete for GROUP=$GROUP"
