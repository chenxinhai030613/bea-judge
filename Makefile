.PHONY: test compile check-env summarize-qlora

PYTHON ?= .venv_qlora/bin/python
PYTHONPYCACHEPREFIX ?= /tmp/bea-judge-pycache
QLORA_SUMMARY_OUTPUT ?= /tmp/bea-judge-qlora-3seed-summary

test:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) -m unittest discover -s tests

compile:
	PYTHONPYCACHEPREFIX=$(PYTHONPYCACHEPREFIX) $(PYTHON) -m compileall -q src scripts tests

check-env:
	$(PYTHON) -c "import pytest, matplotlib, docx"

summarize-qlora:
	$(PYTHON) scripts/summarize_qlora_3seed.py --seeds 13 42 2026 --run-suffix _1024 --allow-missing --output-dir $(QLORA_SUMMARY_OUTPUT)
