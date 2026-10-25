# 🏎️ F1 Race Pace Predictor

[![CI](https://github.com/mallamteja/f1_models/actions/workflows/ci.yml/badge.svg)](https://github.com/mallamteja/f1_models/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Live-f1predictor.tech-FF1801?style=for-the-badge&logo=formula1)](https://f1predictor.tech/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)

A FastAPI service that predicts a driver's race pace (seconds per lap) for four Grand Prix circuits — **Abu Dhabi, Qatar, USA (Austin) and Mexico** — from qualifying time, clean-air race pace, constructor strength and weather inputs. One pre-trained model per circuit is served from `models/`.

---

## Architecture

```
            training (offline, needs network)                     serving (offline, no network)
 ┌───────────────────────────────────────────────┐        ┌──────────────────────────────────────┐
 │ FastF1 2024 race laps ─┐                       │        │ uvicorn main:app                     │
 │ OpenWeather forecast ──┼─► training/*.py ──────┼──────► │  startup: load models/*.joblib       │
 │ hardcoded quali/team ──┘   (one per circuit)   │ .joblib│  POST /predict  → validated input    │
 └───────────────────────────────────────────────┘        │            → feature row → model     │
                                                           │  GET  /health   → 200 / 503          │
                                                           └──────────────────────────────────────┘
```

| Circuit   | Training script       | Artifact                        | Model                       |
|-----------|-----------------------|---------------------------------|-----------------------------|
| Mexico    | `training/6.py`       | `models/mexico_model.joblib`    | GradientBoostingRegressor + SimpleImputer |
| USA       | `training/8-1.py`     | `models/us_model.joblib`        | GradientBoostingRegressor + SimpleImputer |
| Qatar     | `training/10-3.py`    | `models/qatar_model.joblib`     | XGBRegressor (monotone constraints) + SimpleImputer |
| Abu Dhabi | `training/12-3.py`    | `models/abu_dhabi_model.joblib` | XGBRegressor (monotone constraints) |

`training/12.py` and `training/17.py` (Keras FFN → `models/abu_dhabi_ffnmodel.keras`) are experiments; the API does not load their output. `models/shap_*.png` are SHAP feature-importance plots written by the training scripts.

**Repository layout**

| Path | What it is |
|------|------------|
| `main.py`, `config.py` | API and its settings (`PORT`, `MODELS_DIR`) |
| `models/` | Served model artifacts + `lookup_data.json` (driver → team score) |
| `training/` | Training scripts; `train_all.py` runs the four served ones |
| `tests/test_main.py` | pytest suite (`inferencetest.py` / `throughputtest.py` are manual load scripts against a running server) |
| `datasets/`, `visulizations&reports/` | Season data, exploration notebooks and Power BI reports — not used by the API |

---

## Setup

Requires Python 3.11 (3.12 also works locally).

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # API + test/lint tools
```

> **Dependency pins matter.** `requirements.txt` is the human-edited source; the Docker image installs `requirements.lock`, a fully resolved, hash-verified copy for Linux/Python 3.11 (regenerate command is in its header). The committed pickles only load with `scikit-learn==1.5.2`, `numpy==2.2.6` and `xgboost==3.2.0` (scikit-learn 1.8 fails with `AttributeError`, numpy < 2 cannot read them). Do not bump these without retraining every model. Dependabot is configured to skip them.

## Run the API

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI is at <http://localhost:8000/>.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"race_name":"usa","driver_code":"VER","qualifying_time":94.5,"clean_air_race_pace":100.2,"rain_prob":0,"temperature":35}'
```

```json
{"race":"usa","driver":"VER","predicted_pace":102.09543744732858,"meta":{"latency":"0.0098s","model":"usa_v2"}}
```

Accepted `race_name` aliases: `abudhabi`/`abu_dhabi`/`yas_marina`, `qatar`/`lusail`, `usa`/`united_states`/`austin`/`cota`, `mexico`/`mexico_city`. Lap times must fall inside each circuit's plausible window and race pace must be slower than qualifying, otherwise the API returns `422`.

| Endpoint | Purpose |
|----------|---------|
| `POST /predict` | Race pace prediction |
| `GET /health` | `200` with loaded model list; `503` if no model loaded (used by the Docker healthcheck) |
| `GET /info` | Version and supported races |

## Tests and lint

```bash
pytest
ruff check .
ruff format --check main.py config.py tests/test_main.py
```

The suite runs the real startup path, checks validation rules, and pins each circuit's prediction for a fixed input (tolerance 0.01), so a dependency change that silently alters model output fails CI.

## Training

Training pulls live data from FastF1 and OpenWeather, so it needs network access and an API key.

```bash
pip install -r requirements-train.txt
cp .env.example .env                 # then set openweatherapi=<your key>
python training/train_all.py --dry-run     # show what would run
python training/train_all.py --race usa    # retrain one circuit
python training/train_all.py               # retrain all four
```

⚠️ Training **overwrites** the committed `models/*.joblib`. Results are not bit-for-bit reproducible: FastF1 data and the weather forecast change over time. After retraining, update the expected values in `tests/test_main.py` deliberately.

---

## Docker

The image is multi-stage, runs as a non-root user, contains only `main.py`, `config.py` and the served models, and has a `HEALTHCHECK` on `/health`.

```bash
docker build -t f1_models:local .
docker run --rm -p 8000:8000 f1_models:local
# published image
docker run --rm -p 8000:8000 ghcr.io/mallamteja/f1_models:latest
```

Override the port with `-e PORT=9000 -p 9000:9000`. Models are baked into the image (≈1.5 MB total), so no runtime download or secrets are needed.

## CI/CD

| Workflow | Trigger | Does |
|----------|---------|------|
| `.github/workflows/ci.yml` | PR and push to `main` | ruff, pytest, Docker build (not pushed), container smoke test of `/health` + `/predict` |
| `.github/workflows/publish.yml` | tag `vX.Y.Z` on a commit in `main` | tests, then build and push `ghcr.io/mallamteja/f1_models:<tag>` and `:latest` using `GITHUB_TOKEN` |

Third-party actions are pinned to commit SHAs; Dependabot keeps them current.

### Release checklist

1. `main` is green in CI.
2. Bump `APP_VERSION` in `config.py` if the API changed, commit, push, wait for green. If `requirements.txt` changed, regenerate `requirements.lock` first.
3. Tag the commit on `main` and push (only strict `vX.Y.Z` tags publish, and the workflow refuses commits that are not on `main`):
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. Watch the **Publish** workflow in the Actions tab.
5. The package is public (inherited from the public repo), so anyone can pull it without logging in.
6. Verify: `docker run --rm -p 8000:8000 ghcr.io/mallamteja/f1_models:v0.1.0`, then `curl localhost:8000/health`.

## Deployment

The live site at <https://f1predictor.tech> is hosted on Render using `render.yaml` (`uvicorn main:app`, health check `/health`).

## Known limitations

- `models/us_model.joblib` was pickled with scikit-learn 1.8.0 and loads under 1.5.2 with an `InconsistentVersionWarning`; its output is identical to 1.8.0 on the test input and is pinned by a regression test. Retraining it under 1.5.2 removes the warning.
- Each training set has ~20 drivers; reported MAE comes from ~2 test rows and is not a reliable quality measure.

---

Disclaimer: This project is unofficial and is not associated in any way with the Formula 1 companies. F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks are trademarks of Formula One Licensing B.V.
