# BEA-Judge Public Release Notes

## Release scope

The GitHub repository stores reviewable BEA-Judge source code, experiment
configurations, tests, lightweight metadata, and compact result summaries.
Model weights, runtime environments, large datasets, logs, archives, and
generated manuscript files remain outside the public repository.

## Files intended for GitHub

- `src/`, `scripts/`, `configs/`, and `tests/`
- `README.md`, `Makefile`, `requirements.txt`, and
  `REPRODUCIBILITY_MANIFEST.json`
- Small, non-sensitive summaries under `artifacts/`
- Lightweight dataset metadata under `datasets/academic_research/`

## Files kept local

- `models/`, `judge/`, `_deps/`, virtual environments, and dependency caches
- Raw, processed, split, SFT, judge-output, and model-output datasets
- `paper/`, `archive/`, `logs/`, generated Word/PDF files, and Python caches

These files are not removed from the local workspace. Publishing a dataset or
model requires a separate license, privacy/security review, checksum, and
download guide.

## Pre-push checks

Run the following from the repository root:

```bash
git status --short
git add .
git diff --cached --stat
git diff --cached --check
```

The staged set should contain only source code, configuration, tests,
documentation, metadata, and deliberately selected small summaries. Never
force-add ignored files, access tokens, personal paths, or local logs.

## Reproducibility checks

The minimum validation commands are documented in the main README:

```bash
make check-env
make compile
make test
```

Full training and inference require downloading base models, installing
additional dependencies, and preparing the ignored input datasets. The GitHub
repository does not claim to contain those large runtime resources.
