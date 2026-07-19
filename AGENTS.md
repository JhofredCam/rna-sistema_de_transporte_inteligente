# AGENTS.md

Compact guidance for OpenCode sessions. Complements `README.md` (full command reference and architecture) — read both.

## Layout

- `src/module1_demand/` — LSTM + attention for passenger-demand forecasting. Synthetic dataset in `data/demanda_transporte.csv`.
- `src/module2_distraction/` — Transfer-learning image classifier (mobilenet_v3_small / resnet18 / efficientnet_b0). Needs Kaggle dataset.
- `src/module3_recommender/` — Hybrid neural travel-destination recommender. Needs Kaggle dataset.
- `src/shared/` — base `BaseModel`/`BaseTrainer`/`BaseEvaluator` + `metrics.py`.
- `api/` — FastAPI app (`api/main.py`), routers in `api/routers/`, model singletons in `api/dependencies.py`.
- `web/` — React 19 + Vite + Tailwind frontend.
- `scripts/` — CLI entrypoints for module 2 & 3 (train/eval/predict), `download_data.py`, and `eda_module1_demand.py` (regenerates `docs/figures/module1_demand/`).
- `notebooks/` — EDA notebooks per module (`01_eda_demand`, `02_eda_images`, `03_eda_recommender`). 02 & 03 are executed against `data/raw/` Kaggle data; 01 also contains the training pipeline (legacy).
- `docs/figures/` — EDA figures + `eda_summary.json` per module, consumed by `docs/ReporteTecnico.md`. Regenerate module 1 with `python scripts/eda_module1_demand.py`.
- `models/` — trained artifacts. `models/*` is gitignored except specific files under `models/demand/`, `models/module2_distraction/`, `models/module3_recommender/` (see `.gitignore`). Demand scalers/encoders ARE committed and required by the API.
- `tests/` — `unit/`, `integration/` (currently empty), `e2e/` (real API tests).

## Commands

Python env (Windows / PowerShell — use `.venv\Scripts\activate`, not `source .venv/bin/activate`):
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r api/requirements.txt
```

Backend, from repo root:
```
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend, separate terminal:
```
cd web; npm install; npm run dev     # http://localhost:5173
cd web; npm run lint                 # ESLint (only lint command in the repo)
```
Vite proxies `/demand`, `/distraction`, `/recommender` to `http://localhost:8000` (`web/vite.config.js`). Set `VITE_API_URL` only when the API is not at that origin.

Tests — `pytest.ini` sets `pythonpath = .`, so run from repo root:
```
pytest                                                            # all
pytest tests/unit                                                 # no model artifacts needed
pytest tests/e2e                                                  # requires model artifacts (see below)
pytest tests/e2e/test_api_e2e.py::TestDemand::test_predict_with_history   # single test
```

## Gotchas

- **No Python lint / typecheck / formatter is configured.** Don't assume ruff, black, mypy, or flake8. Only `web/` has ESLint. If you add Python tooling, record the command here.
- **e2e tests require committed model artifacts.** `api/dependencies.py` loads demand + distraction models as singletons on first request. `TestDemand::test_predict_with_history` and `TestDistraction::test_health` will fail if `models/demand/best_model.pth` (+ scalers/encoders) or `models/module2_distraction/best_model.pth` are missing. The recommender endpoint tolerates a missing checkpoint (returns `status: "unavailable"` / 503).
- **`src/module1_demand/train.py` uses bare intra-package imports** (`from data_generator import ...`) — run it as a script (`python src/module1_demand/train.py`), not as a module. It regenerates the synthetic dataset and **overwrites** `data/demanda_transporte.csv` AND `web/public/data/demanda_transporte.csv`. Do not run it casually.
- **Module 2 & 3 training entrypoints live in `scripts/`** (not `src/`) and use package imports (`from src.module2_distraction import ...`). They require Kaggle data first:
  ```
  python scripts/download_data.py --module module2 --output-dir data/raw
  python scripts/download_data.py --module module3 --output-dir data/raw
  ```
  Requires `kaggle` CLI + `%USERPROFILE%\.kaggle\kaggle.json` (see `scripts/setup_kaggle.md`).
- **CORS origins are hardcoded** in `api/main.py`: the Netlify frontend + `localhost:5173`/`5174`. Adding a frontend origin means editing that list.
- **`src/config.py` and `.env.example` are empty.** No env vars are needed for local dev beyond optional `VITE_API_URL` (frontend) and `PORT` (deploy).
- **API routers use Pydantic v2** with `Field(..., ge=, le=)` bounds → 422 on bad values; cross-field validation in routes returns 400.
- **`tmp_test/` is scratch/debug space**, not part of the application.

## Conventions

- Comments, docstrings, docs, and commit messages are in **Spanish** (Colombian academic context). Match this when adding content.
- API deploys via `Dockerfile` / `railpack.json` (Railway, `PORT` env); frontend on Netlify (`sistema-transporte-inteligente-rna.netlify.app`).
