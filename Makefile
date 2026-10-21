PY ?= python3
export PYTHONPATH := src

.PHONY: check lint test figures clean

check: lint test figures

lint:
	ruff check src tests
	ruff format --check src tests

# Unit tests: filter algebra, synthetic trace with known truth, loader, baselines, figure smoke.
test:
	$(PY) -m pytest -q

# Runs the filter on the real day of data, writes figures/*.png and figures/results.json.
figures:
	$(PY) -m kalman1d

clean:
	rm -rf .pytest_cache .ruff_cache src/kalman1d/__pycache__ tests/__pycache__
