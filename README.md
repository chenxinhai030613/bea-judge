# BEA-Judge

BEA-Judge is a four-module framework for bias-aware, evidence-enhanced
evaluation and calibration of large language model (LLM) judgments. This
repository contains the source code, experiment configurations, tests,
reproducibility utilities, and compact result summaries needed to inspect and
extend the project.

## Repository scope

This is a source-first public release. The repository intentionally excludes
model weights, virtual environments, dependency caches, raw and intermediate
datasets, full model outputs, training logs, archives, and generated Word/PDF
manuscripts. Those resources remain local and are covered by `.gitignore`.

The tracked release includes:

- `src/`: BEA-Judge modules and training utilities
- `scripts/`: dataset, experiment, audit, and reporting tools
- `configs/`: reproducible experiment configurations
- `tests/`: unit and integration tests
- `artifacts/`: small result summaries and audit reports
- `datasets/academic_research/`: lightweight dataset metadata

See [`docs/public-release.md`](docs/public-release.md) for the release
boundary and pre-push checks.

## Quick start

Run these commands from the repository root:

```bash
make check-env
make compile
make test
```

The minimum environment uses Python, `numpy`, `pandas`, `matplotlib`,
`python-docx`, and `pytest`. Full QLoRA training and inference additionally
require `torch`, `transformers`, `datasets`, `peft`, `bitsandbytes`,
`accelerate`, and `safetensors`, together with locally available model and
dataset inputs.

## Reproducibility workflow

The public snapshot provides the experiment protocol and the commands below.
Large inputs and outputs must be prepared locally before running them.

### Build pairwise SFT data

```bash
python scripts/build_judge_sft_dataset.py \\
  --config configs/qlora_judge_sft_24gb_epoch1_1024.json
```

### Run the QLoRA experiment

```bash
SEED=13 RUN_SUFFIX=_1024 \\
CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \\
bash scripts/run_qlora_epoch1_seed.sh
```

For the three-seed protocol:

```bash
SEEDS="13 42 2026" RUN_SUFFIX=_1024 RUN_TIE_SENSITIVE=1 \\
CONFIG=configs/qlora_judge_sft_24gb_epoch1_1024.json \\
bash scripts/run_qlora_3seed_epoch1.sh
```

Validate the resulting submission summary with:

```bash
python scripts/validate_qlora_submission_package.py \\
  --submission-summary \\
  datasets/model_outputs/qlora_3seed_epoch1_1024_summary/qlora_submission_ready_results.json
```

### Run audits and diagnostics

The repository includes dedicated tools for:

- bias-awareness auditing: `scripts/bias_awareness_audit.py`
- evidence fact auditing: `scripts/evidence_fact_audit.py`
- order-swap probing: `scripts/order_swap_probe.py`
- tie-sensitive validation: `scripts/build_tie_sensitive_validation_reports.py`
- accuracy-constrained tie rescue: `scripts/accuracy_constrained_tie_rescue_audit.py`

Use explicit input and output paths when running audits so that frozen results
are not confused with exploratory outputs.

## Result interpretation

Compact reports under `artifacts/` are included for inspection. Complete
datasets, model outputs, checkpoints, and generated figures are intentionally
not part of this public repository. In particular, a conservative multi-seed
summary should not be treated as a final submission package unless its gate
reports pass. Accuracy-oriented and tie-sensitive operating points should be
reported separately.

## Naming and generated documents

All tracked paths use English names. Local report and manuscript-generation
scripts write to the ignored `paper/` directory, so generated documents are
not uploaded to GitHub.

## License and responsible use

No license is granted by this repository unless a separate license file is
added. Before redistributing datasets or model artifacts, verify their source
licenses, privacy constraints, and redistribution permissions. Results should
be used as evaluation evidence, not as the sole basis for high-impact
decisions about people.
