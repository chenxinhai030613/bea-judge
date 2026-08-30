#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/ubuntu/bea-judge/1论文写作}"
VENV_DIR="${VENV_DIR:-.venv_qlora}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
RUN_PILOT="${RUN_PILOT:-1}"

echo "[0/6] Entering project: $PROJECT_DIR"
cd "$PROJECT_DIR"
pwd
ls configs scripts src datasets models >/dev/null
test -f scripts/run_qlora_pilot.sh

echo "[1/6] Preparing Linux virtual environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if [ "$INSTALL_DEPS" = "1" ]; then
  echo "[2/6] Installing QLoRA dependencies"
  pip install torch transformers accelerate datasets peft bitsandbytes trl safetensors scikit-learn scipy numpy
else
  echo "[2/6] Skipping dependency installation because INSTALL_DEPS=$INSTALL_DEPS"
fi

echo "[3/6] Checking GPU and QLoRA Python dependencies"
python - <<'PY'
import importlib.util

required = ("torch", "transformers", "accelerate", "datasets", "peft", "bitsandbytes")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing dependencies: {', '.join(missing)}")

import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available; do not run QLoRA on CPU.")

props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU: {props.name}; VRAM_GB={vram_gb:.2f}")
if vram_gb < 20:
    raise SystemExit("Detected VRAM < 20GB. Use the fallback config before running this pilot.")
PY

echo "[4/6] Checking required experiment inputs"
test -f datasets/processed/bea_judge_cleaned_10000.json
test -d models/M-Prometheus-3B
test -f configs/qlora_judge_sft.json
test -f datasets/model_outputs/bea_judge_20260521_110114/validation_report.json
test -f datasets/judge_outputs/m_prometheus_3b_bea10k_v2/summary.json

echo "[5/6] Checking pilot shell syntax"
bash -n scripts/run_qlora_pilot.sh

if [ "$RUN_PILOT" != "1" ]; then
  echo "[6/6] Preflight complete. Skipping pilot because RUN_PILOT=$RUN_PILOT"
  exit 0
fi

echo "[6/6] Starting seed42 QLoRA pilot"
bash scripts/run_qlora_pilot.sh

