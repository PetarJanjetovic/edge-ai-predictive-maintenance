.PHONY: setup data train dashboard simulate test clean

PYTHON := .venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

data:
	$(PYTHON) -m edge_pdm.cli generate --samples-per-class 500

train:
	$(PYTHON) -m edge_pdm.cli train

dashboard:
	$(PYTHON) -m edge_pdm.cli dashboard

simulate:
	$(PYTHON) -m edge_pdm.cli simulate --url http://127.0.0.1:8000/api/telemetry

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf .pytest_cache build src/*.egg-info

