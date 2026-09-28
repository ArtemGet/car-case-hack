.PHONY: setup test lint weight-gate docker

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check .

weight-gate:
	$(PYTHON) tools/weight_gate.py --dir . --include-dir artifacts

docker:
	docker compose config
	docker compose build
