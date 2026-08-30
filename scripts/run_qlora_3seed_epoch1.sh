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

SEEDS="${SEEDS:-13 42 2026}"
RUN_SUFFIX="${RUN_SUFFIX:-_1024}"
CONFIG="${CONFIG:-configs/qlora_judge_sft_24gb_epoch1_1024.json}"
SUMMARY_DIR="${SUMMARY_DIR:-datasets/model_outputs/qlora_3seed_epoch1${RUN_SUFFIX}_summary}"
RUN_TIE_SENSITIVE="${RUN_TIE_SENSITIVE:-1}"
TIE_SENSITIVE_SUFFIX="${TIE_SENSITIVE_SUFFIX:-${RUN_SUFFIX}_tie_sensitive_dev}"
TIE_SENSITIVE_SUMMARY_DIR="${TIE_SENSITIVE_SUMMARY_DIR:-datasets/model_outputs/qlora_3seed_epoch1${TIE_SENSITIVE_SUFFIX}_summary}"
FROZEN_REPORT="${FROZEN_REPORT:-datasets/model_outputs/bea_judge_20260521_110114/validation_report.json}"
RAW_FROZEN_SUMMARY="${RAW_FROZEN_SUMMARY:-datasets/judge_outputs/m_prometheus_3b_bea10k_v2/summary.json}"

echo "[3seed] Running epoch1 protocol for seeds: $SEEDS"
echo "[3seed] RUN_SUFFIX=$RUN_SUFFIX"
echo "[3seed] CONFIG=$CONFIG"
echo "[3seed] RUN_TIE_SENSITIVE=$RUN_TIE_SENSITIVE"
for seed in $SEEDS; do
  echo "[3seed] Starting seed $seed"
  SEED="$seed" RUN_SUFFIX="$RUN_SUFFIX" CONFIG="$CONFIG" bash scripts/run_qlora_epoch1_seed.sh
done

echo "[3seed] Summarizing comparison reports"
comparison_dirs=()
for seed in $SEEDS; do
  comparison_dirs+=("datasets/model_outputs/qlora_comparison_seed${seed}_epoch1${RUN_SUFFIX}")
done
"$PYTHON" scripts/summarize_qlora_3seed.py \
  --seeds $SEEDS \
  --comparison-dirs "${comparison_dirs[@]}" \
  --output-dir "$SUMMARY_DIR"

if [ "$RUN_TIE_SENSITIVE" = "1" ]; then
  echo "[3seed] Building dev-selected tie-sensitive validation reports"
  "$PYTHON" scripts/build_tie_sensitive_validation_reports.py \
    --seeds $SEEDS \
    --validation-template "datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1${RUN_SUFFIX}/validation_report.json" \
    --calibrated-template "datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1${RUN_SUFFIX}/calibrated_results.json" \
    --output-template "datasets/model_outputs/bea_judge_qlora_pairwise_seed{seed}_epoch1${TIE_SENSITIVE_SUFFIX}"

  echo "[3seed] Writing tie-sensitive comparison reports"
  tie_comparison_dirs=()
  for seed in $SEEDS; do
    tie_dir="datasets/model_outputs/qlora_comparison_seed${seed}_epoch1${TIE_SENSITIVE_SUFFIX}"
    tie_comparison_dirs+=("$tie_dir")
    "$PYTHON" scripts/compare_qlora_experiments.py \
      --frozen-report "$FROZEN_REPORT" \
      --qlora-report "datasets/model_outputs/bea_judge_qlora_pairwise_seed${seed}_epoch1${TIE_SENSITIVE_SUFFIX}/validation_report.json" \
      --raw-frozen-summary "$RAW_FROZEN_SUMMARY" \
      --raw-qlora-scores "datasets/judge_outputs/m_prometheus_3b_qlora_pairwise_seed${seed}_epoch1${RUN_SUFFIX}/base_scores.json" \
      --output-dir "$tie_dir"
  done

  echo "[3seed] Summarizing tie-sensitive comparison reports"
  "$PYTHON" scripts/summarize_qlora_3seed.py \
    --seeds $SEEDS \
    --comparison-dirs "${tie_comparison_dirs[@]}" \
    --output-dir "$TIE_SENSITIVE_SUMMARY_DIR"

  echo "[3seed] Writing submission-ready dual-operating-point summary"
  "$PYTHON" scripts/build_qlora_submission_summary.py \
    --conservative-summary "$SUMMARY_DIR/three_seed_summary.json" \
    --tie-sensitive-summary "$TIE_SENSITIVE_SUMMARY_DIR/three_seed_summary.json" \
    --output-dir "$SUMMARY_DIR"

  echo "[3seed] Validating submission-ready dual-operating-point package"
  "$PYTHON" scripts/validate_qlora_submission_package.py \
    --submission-summary "$SUMMARY_DIR/qlora_submission_ready_results.json"
fi

echo "[3seed] Complete. Inspect $SUMMARY_DIR/qlora_submission_ready_results.md"
