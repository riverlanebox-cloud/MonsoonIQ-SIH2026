.PHONY: setup data grid train evaluate console build-ui smoke serve api ui test all clean reset

PYTHON := .venv/bin/python
PIP := .venv/bin/pip
PYTEST := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn

# ---------------------------------------------------------------------------
# Full local run, in order. Times measured on a 4-core sandbox, warm pip cache:
#
#   setup     ~2 min   python deps + npm install
#   data      ~65 s    district archive (2,088 days x 53 districts)
#   grid      ~5 s     re-simulates the grid archive (needed for the FSS exhibit)
#   train     ~40 s    6 stages
#   evaluate  ~20 s    verification artifacts + PDF
#   console   ~20 s    season timeline + ranked significant days
#   build-ui  ~5 s     production bundle served by the API
#   test      ~6 s     34 pytest tests
#
# Reproducibility: within one environment the generator is bit-identical run to
# run (verified by hashing the parquet across repeated runs). Across
# environments, NumPy/BLAS version differences move the archive by ~1e-13, so
# `make grid` gates on a 1e-4 tolerance and every evaluation writes the archive
# fingerprint (sha256, first 16 hex) next to its results. Re-running the whole
# chain after a fresh `make data` reproduced all 1,646 published fields exactly.
# ---------------------------------------------------------------------------

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	cd frontend && npm install

# District-scale archive. Deterministic: same seed, same parquet.
data:
	PYTHONPATH=. $(PYTHON) -m src.data.synthetic_generator

# Grid archive used by the spatial (FSS) verification and the map's grid view.
# Re-simulates the generator with the same seed and refuses to write unless the
# overlapping dates agree with the committed sample - that gate is what makes
# the archive auditable. Skip it and evaluation reports grid FSS as unavailable.
grid:
	PYTHONPATH=. $(PYTHON) scripts/build_grid_samples.py

train:
	PYTHONPATH=. $(PYTHON) src/train.py

evaluate:
	PYTHONPATH=. $(PYTHON) src/evaluate.py

# Precomputed console payloads: the season timeline and the ranked significant
# days. Without this the Today screen still works (live inference per date) but
# the timeline scrubber and its jump list are empty.
console:
	PYTHONPATH=. $(PYTHON) -m src.console.artifacts

# Production bundle of the console, served by the API at / (one process, no CORS).
build-ui:
	cd frontend && npm run build

smoke:
	cd frontend && npm run smoke

test:
	PYTHONPATH=. $(PYTEST) tests/ -q

# Everything needed before a demo, cheapest ordering.
all: data grid train evaluate console build-ui test

# Same-origin console + API on :8000 (binds 0.0.0.0 so a phone on the LAN or a
# container can reach it too).
serve:
	PYTHONPATH=. $(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000

# Backend with autoreload, and the UI on :3000 with hot reload. Run in two
# terminals; Vite proxies /api to :8000.
api:
	PYTHONPATH=. $(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	cd frontend && npm run dev

# Re-record the API fixtures the console smoke test renders against.
fixtures:
	PYTHONPATH=. $(PYTHON) scripts/dump_api_fixtures.py --date 2020-08-05 --lead 1

clean:
	rm -rf __pycache__ .pytest_cache src/**/__pycache__ tests/__pycache__
	rm -f artifacts/models/*.joblib artifacts/plots/*.png artifacts/reports/*.pdf
	rm -f artifacts/console/*.json
	rm -rf frontend/dist frontend/.smoke-build

# Regenerate the archives from scratch (data + artifacts), then rebuild.
reset: clean all
