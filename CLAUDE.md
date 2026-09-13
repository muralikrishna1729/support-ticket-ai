# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Project Overview

**SmartTicket AI** — an end-to-end, asynchronous ML system that classifies customer support tickets. Users submit a ticket, get a ticket ID back immediately, and the ML engine (in the background) assigns a **category**, an **issue type**, a **confidence score**, and an **auto-response**. The worker is designed to scale horizontally.

The full lifecycle is covered: EDA notebooks → training pipeline → model sync to S3 → FastAPI ingestion → background classification → Streamlit dashboard → AWS monitoring/alerts → Docker/CI-CD deployment.

## Architecture

```
Streamlit UI ──> FastAPI (src/app.py) ──> PostgreSQL (SQLAlchemy)
                     │
                     ├── in-process: FastAPI BackgroundTasks (default path)
                     │        └── PredictPipeline → DB update
                     └── queue: AWS SQS (worker.py polls) ──> PredictPipeline → DB update
                                                              │
Model artifacts (TF-IDF + 2× LinearSVC) loaded from ./models/  ├── AWS CloudWatch metrics
  — downloaded from S3 on startup if missing                   └── AWS SES failure email
```

- **Two processing paths exist** and both call `process_ticket_background()`:
  1. **In-process** — `POST /tickets` enqueues `process_ticket_background` via FastAPI `BackgroundTasks` (`src/api/routes.py:19`). This is what actually runs today.
  2. **SQS worker** — `python worker.py` polls SQS and processes ticket IDs from queue messages. Note: **nothing currently publishes to SQS** — `aws/sqs_client.py` defines `send_message()` but it is not called anywhere. The worker is wired up but idle unless a publisher is added.

## Tech Stack

| Layer | Tech |
| --- | --- |
| Frontend | Streamlit 1.39 (`streamlit_app.py`), Plotly, dark custom CSS (Space Grotesk / JetBrains Mono) |
| API | FastAPI 0.115, Uvicorn, Pydantic v2 |
| DB | PostgreSQL 16 (Docker) / SQLite fallback (dev, `DATABASE_URL` unset) via SQLAlchemy 2.0 |
| ML | scikit-learn: TF-IDF vectorizer + LinearSVC (multi-output: category & issue type), LabelEncoders, `compute_class_weight` |
| AWS | boto3 — S3 (model storage), SQS (queue), SES (failure emails), CloudWatch (custom metrics) |
| Deploy | Docker / docker-compose, nginx reverse proxy, GitHub Actions (CI/CD to EC2) |

## Getting Started / Running

```bash
# Local dev (SQLite fallback — no DB or AWS needed to boot the API)
uvicorn src.app:app --reload          # API on :8000 (docs at /docs)
python worker.py                      # SQS worker (optional)
streamlit run streamlit_app.py        # UI on :8501

# Tests
pytest tests/ -v                      # requires TESTING=true (see below)

# Docker (dev: postgres + api + worker)
docker compose -f docker-compose.yml up --build

# Docker (prod: api + worker + frontend + nginx on :80)
docker compose -f docker-compose.prod.yml up --build
```

> ⚠️ There is **no `main.py`** — the app entry point is `src.app:app`. The README's `uvicorn main:app` is outdated.

## Project Structure

```
aws/                      # boto3 clients (s3, sqs, ses, cloudwatch)
src/
  app.py                  # FastAPI app factory (entry point: src.app:app)
  api/routes.py           # REST endpoints (/tickets, /tickets/search, /tickets/{id}, /analytics/summary)
  components/             # data_ingestion, data_transformation, model_trainer
  db/                     # database.py (engine/session), models.py (Ticket), schemas.py (Pydantic)
  pipeline/               # train_pipeline.py, predict_pipeline.py
  services/ticket_service.py  # business logic: create/process/search/analytics
  utils.py                # save/load objects, JSON, evaluate_model
  logger.py               # logging (console + timestamped file in ./logs/)
  exception.py            # CustomException
scripts/upload_models.py  # uploads ./models/*.pkl to S3
notebook/                 # EDA + data CSVs (2.ModelTraining.ipynb is empty/0 bytes)
nginx/nginx.conf          # routes / → frontend:8501, /api/ → api:8000
streamlit_app.py          # UI: Classify / Search / Analytics pages
worker.py                 # SQS polling loop
```

## Core Flows

### Ticket lifecycle
`pending` → `processing` → `completed` | `failed` (set by `src/services/ticket_service.py`).

`POST /tickets` (body `{"ticket": "..."}`) creates a DB row with `status="pending"`, then schedules `process_ticket_background(id)`. On success the row is filled with category/issue_type/auto_response/confidence/needs_review and marked `completed`; on failure it is marked `failed`, a CloudWatch `PredictionError` metric is logged, and an SES alert email is sent.

### Prediction (`src/pipeline/predict_pipeline.py`)
1. `DataTransformation.clean_text()` — lowercase, strip non-letters, drop stopwords (EN + custom) and words ≤2 chars.
2. TF-IDF transform → category LinearSVC predict → label-inverse → issue-type LinearSVC predict.
3. `confidence` = max of category decision-function scores; `needs_review = confidence < 0.60`.
4. Auto-response from the `RESPONSES` dict (10 known categories); incidents get a 🚨 prefix.

### Model bootstrap (`ensure_models_exist`)
At `PredictPipeline()` init, the 5 artifacts in `models/` are loaded; any missing files are downloaded from S3 (`models/<name>` key) and the process **raises if any download fails** (worker/API container will fail on start — intentional).

## Model Artifacts & Performance

Models live in `./models/` (gitignored, backed up to S3):

| File | Contents |
| --- | --- |
| `tfidf_vectorizer.pkl` | TF-IDF (max_features=20000, ngram 1–3, sublinear) |
| `clf_category.pkl` | LinearSVC, GridSearchCV over C (best **C=2.0**), `class_weight='balanced'` |
| `clf_issue_type.pkl` | LinearSVC, C=1.0, `class_weight='balanced'` |
| `le_category.pkl` / `le_issue_type.pkl` | LabelEncoders |
| `model_scores.json` | Current scores (weighted F1): **category 0.5286, issue_type 0.8063** |

> The README claims 0.68 / 0.89 — these are stale. `model_scores.json` is the source of truth.

Current model version: **v3.0-negation-tfidf** — aggressive cleaning with negation tagging
(`cannot access` → `not NEG_access`; cue words kept, next 3 content words get a `NEG_`
prefix) feeding the same TF-IDF + LinearSVC stack. `diagnose_taxonomy.py` handles the
confusion-matrix / label-noise analysis (results in `taxonomy_results.json`), and
`benchmark_techniques.py` holds the feature-technique bake-off (`experiment_results.json`).

### Training
```bash
python -m src.pipeline.train_pipeline    # uses hardcoded notebook/data/dataset-tickets-multi-lang-4-20k.csv
# then sync to S3:
python scripts/upload_models.py
```
> Note: `--source_path` (as shown in the README) is **not actually parsed** by the CLI — the `__main__` block hardcodes the source file. The README is stale here.

## Environment Variables

Copy `.env.example` → `.env`. Key vars:

- `DATABASE_URL` — default `sqlite:///./tickets.db`; Docker sets a Postgres URL (`postgresql://ticketuser:ticketpass@db:5432/smartticket`, host port **5433**).
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` (region is `ap-south-1`)
- `SQS_QUEUE_URL`, `S3_BUCKET_NAME`, `SES_SENDER_EMAIL`, `SES_RECEIVER_EMAIL`
- `API_URL` — base URL the Streamlit app calls (default `http://localhost:8000`)
- `TESTING` — when `"true"`, `PredictPipeline` skips all model loading (used by CI).

## Testing

- `tests/test_api.py` — health/home endpoints; an `autouse` fixture patches `src.services.ticket_service.pipeline` so no models load.
- `tests/test_predict.py` — `clean_text()` unit tests.
- CI runs `pytest tests/` with `TESTING=true` and dummy AWS env vars. Set `TESTING=true` locally too or the pipeline will try to download models from S3.

## CI/CD & Deployment

`.github/workflows/deploy.yml`:
- **test** job: Python 3.11, `pip install -r requirements.prod.txt`, `pytest tests/`.
- **deploy** job (on push to `main`, after tests pass): SSHes to EC2 (`secrets.EC2_HOST/USERNAME/SSH_KEY`), `git pull`, rebuilds and restarts `docker-compose.prod.yml`, then `curl -f http://localhost/api/health`.

## Known Quirks & Gotchas

- **`requirements.txt` is UTF-16 encoded** (with BOM/CRLF) — probably a Windows/PowerShell accident. `pip install -r requirements.txt` and `setup.py`'s `get_requirements()` will produce mangled package names (`\x00`-interleaved). The production path uses `requirements.prod.txt` (plain UTF-8) — prefer it.
- **Dependency pinning pain**: `requirements.prod.txt` pins Streamlit to 1.39.0, pandas to 2.2.3, uvicorn to 0.32.1, etc. to resolve Starlette/import conflicts — do not casually bump these; commit history is full of such fixes.
- **Docker build must not use `--no-cache`** (see git history / recent commits) and the health check needs `curl` (installed via apt in the Dockerfile).
- `src/exception.py` does `from src.logger import logging` (stdlib logging re-exported) and imports `error` from `copy` (unused). Logger name is `"smartTicekt"` (typo, harmless).
- `needs_review` threshold (`0.60`) and confidence = `max()` of decision scores can go **negative** for low-confidence classes — expected for LinearSVC raw scores, not a bug.
- `test_rds_connection.py` contains a hardcoded RDS endpoint + password committed to the repo — a one-off debug script; do not reuse/expand it.
- `.env` is gitignored; `.env.example` is the template.
- The SQS worker and the API in-process path can both process tickets — be deliberate about which one is running in an environment to avoid double-processing.
