# SmartTicket AI — Improvement Plan (portfolio makeover)

> **Status:** Plan saved on 2026-08-03. Not yet implemented. We'll work through this phase-by-phase when we resume coding. Each phase is independently committable, so we can stop after any phase.
>
> **User decisions that shape this plan:** all four improvement areas are equal priority; **maximum effort** (full makeover); **upgrade the classical sklearn stack** (no transformer/GPU); **local-first** — the AWS free tier has expired, so the entire demo must run with **zero AWS** (SQLite + local MLflow + local queue), while AWS code stays intact as an optional production path to revisit later.

## Context

SmartTicket AI is an end-to-end async ML system that classifies customer-support tickets (FastAPI + Streamlit + SQLAlchemy/Postgres + sklearn LinearSVC + AWS). It's the owner's fresher portfolio piece. It works, but has four weaknesses this plan fixes:

1. **Weak model** — category F1 is only **0.54**; confidence scores are raw `max(decision_function)` (can go negative) with a hardcoded 0.60 threshold.
2. **No MLOps** — no experiment tracking, no migrations (`create_all` at import), no structured logging, no model versioning.
3. **Half-built async** — SQS `send_message()` is defined but never called; only the in-process FastAPI `BackgroundTasks` path actually runs.
4. **Repo hygiene** — `requirements.txt` is UTF-16 (broken for pip), a 0-byte notebook is tracked, `test_rds_connection.py` has a committed live RDS endpoint + plaintext password, and the README is stale (wrong entry point, outdated metrics).

The result should be a coherent story an interviewer can walk through: *schema lifecycle, observability, experiment tracking, model versioning, a measurably better model, human-in-the-loop feedback, interpretability, a real decoupled queue, hardened CI, and clean docs.*

## Phase summary

| # | Phase | Resume story | Depends on |
|---|---|---|---|
| 0 | Security + zero-AWS boot | "Repo is secret-clean and demos with no cloud" | — |
| 1 | Migrations + health + structured logging | "Owned schema lifecycle; built an observable service" | 0 |
| 2 | MLflow tracking + model registry | "Trained, versioned, and registered models" | 1 |
| 3 | Ensemble + calibration + Optuna | "Improved category F1 0.54 → 0.63+" | 2 |
| 4 | Human feedback loop | "Closed the human-in-the-loop → retraining data" | 1 |
| 5 | Explainability panel | "Showed *why* a ticket was categorized" | 3, 1 |
| 6 | Pluggable queue (local + SQS) | "Built a decoupled async architecture, demoable offline" | 0 |
| 7 | CI/CD hardening | "Production-hardened CI: lint, audit, secret scan" | all |
| 8 | Repo hygiene + docs | "A repo that looks like real OSS" | all |

---

## Phase 0 — Security + zero-AWS boot

**Goal:** remove committed secrets/artifacts; app boots with no AWS; fix broken dependency files.

- **Delete** `test_rds_connection.py` (live RDS endpoint + plaintext password).
- **Rotate AWS keys** in `.env` (free tier is over; keys will be unused). Advise: do NOT push `.env`; old keys recoverable from history only on the closed-source remote — optional `git-filter-repo` rewrite deferred.
- **`.gitignore`** — add `tickets.db`, `mlruns/`.
- **`src/pipeline/predict_pipeline.py`** — `ensure_models_exist()`: when no S3 config present, log a warning and continue with whatever is in `models/` instead of **raising** (today it raises → app won't boot without AWS). This is the critical local-demo fix.
- **`requirements.txt`** — rewrite as **UTF-8** from `pip freeze` so `pip install -r requirements.txt` and `setup.py` work.
- **`requirements.prod.txt`** — accept the working-tree `plotly==6.8.0` change into the commit.

**Verify:** `uvicorn src.app:app` boots with empty/absent AWS env vars; `GET /health` 200; `file requirements.txt` shows ASCII/UTF-8.

## Phase 1 — Alembic migrations + deeper /health + structured logging

**Goal:** schema lifecycle via migrations (replaces `create_all` at import); observable service.

- **Add `alembic==1.13.3`** to `requirements.prod.txt` (new `# MLOps` section).
- **New** `alembic.ini`, `alembic/env.py` (reads `DATABASE_URL`, `render_as_batch=True` for SQLite ALTER, imports `Base.metadata`), `alembic/versions/0001_initial.py` (tickets table + indexes as today), `alembic/versions/0002_*.py` (adds the four future columns now, so later phases never touch schema):
  `model_version: String(50) index`, `feedback: String(20)`, `explanation: Text`, `predicted_text: Text`.
- **`src/app.py`** — remove `Base.metadata.create_all()` at import; run migrations at startup instead.
- **`src/db/database.py`** — `run_migrations()` (subprocess `alembic upgrade head`), skipped when `CHECK_DB_SCHEMA=false` (tests).
- **`src/app.py` `/health`** — return `{status, checks:{database:{ok,detail}, models:{ok,detail}}}`; model check returns `ok` when `TESTING=true` (keeps existing tests green).
- **`src/logger.py`** — rewrite with `python-json-logger` + a `request_id` (uuid4) `Filter`; JSON console formatter; keep timestamped file handler. No call-site changes.
- **`src/api/routes.py`** — middleware sets `request_id` contextvar; `X-Request-ID` header on responses; exception handler logs JSON.

**Verify:** `alembic upgrade head` on fresh SQLite → `0001`+`0002` apply; `uvicorn` boots and `/health` shows both checks; request logs are JSON with `request_id`; `TESTING=true pytest` green.

## Phase 2 — MLflow experiment tracking + model registry

**Goal:** every training run logged (params/metrics/artifacts) and registered as a versioned model; production records which version made each prediction.

> **Front-load a compatibility spike first:** install `mlflow==2.19.0` and smoke-test import + a 1-run tracking call against the pinned venv (scipy 1.17.1, streamlit 1.39, pandas 2.2.3). MLflow pulls flask/opentelemetry; this is the single likeliest break. If it fails, pin a compatible MLflow (e.g. 2.18.x) or run MLflow in its own venv/service before proceeding.

- **Add `mlflow==2.19.0`**.
- **New `src/tracking.py`** — `get_mlflow_tracking_uri()` (env `MLFLOW_TRACKING_URI`, default `sqlite:///./mlruns/mlflow.db`), `start_run()` ctx manager, `register_latest_artifacts()` → registers `CategoryModel`/`IssueTypeModel`, `resolve_model_version()` (env override → staging → production → latest).
- **`src/pipeline/train_pipeline.py`** — wrap run in `start_run`; log params/metrics/artifacts; register versions; add `--experiment` arg.
- **`src/pipeline/predict_pipeline.py`** — set `self.model_version` at init (`"local"` if no MLflow DB); `predict()` returns `model_version`.
- **`src/services/ticket_service.py`** — store `result["model_version"]` on the ticket.
- **`src/db/schemas.py`** — `TicketResponse` gains `model_version: Optional[str]`.
- **`src/api/routes.py`** — new `GET /models/info` (current versions + tracking URI), `GET /experiments` (runs + best F1).
- **`docker-compose.yml`** — add local `mlflow` service (`ghcr.io/mlflow/mlflow:v2.19.0`, port 5000, sqlite store, volume). Training works without it; the server is for browsing.
- **`.env.example`** — `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_VERSION=`.

**Verify:** run training → run appears in `mlruns/`, models registered; `GET /models/info` reports versions; a ticket's `model_version` flows to the API response.

## Phase 3 — Model quality: ensemble + calibration + Optuna

**Goal:** lift category F1 (0.54 → target 0.63+) with proper [0,1] confidence. Classical stack only.

- **Add `optuna==3.6.1`**, `imbalanced-learn==0.14.2` (explicit), **keep `scikit-learn` pinned**.
- **`src/components/model_trainer.py`** — rewrite:
  ```
  VotingClassifier(estimators=[
    ("svc", CalibratedClassifierCV(LinearSVC(C=?, max_iter=3000, class_weight=dict), cv=3, method="sigmoid")),
    ("lr",  LogisticRegression(C=?, max_iter=3000, class_weight="balanced", solver="liblinear")),
    ("sgd", SGDClassifier(loss="log_loss", alpha=?, max_iter=3000, class_weight="balanced")),
  ], voting="soft")
  ```
  Soft voting needs `predict_proba` everywhere → SVC via calibration, sgd via native `log_loss` probabilities.
- **Optuna** per target: log-uniform `C`/`alpha`, `penalty` for sgd, 5-fold stratified CV maximizing `f1_weighted`; ~30–50 trials, `TPESampler(seed=42)`, `n_jobs=2` (memory).
- **`src/pipeline/predict_pipeline.py`** — confidence = `float(np.max(predict_proba))` (real probability, fixes negative-confidence quirk); `REVIEW_THRESHOLD = 0.60` constant (exposed to `/health` + UI), `needs_review = confidence < REVIEW_THRESHOLD`. `clean_text` unchanged.
- **Keep artifacts atomic**: write all 7 pkls to temp names then rename, so a stale partial set never breaks `predict_proba`.
- **Language decision:** stay **EN-only** (current cleaning strips non-ASCII, so multi-lang just adds noise). Keep the `language` filter. Per-language models noted as future work, not this phase.

**Verify:** retrain → `model_scores.json` updated (compare vs 0.5366/0.8087 baseline); `/health` and UI show the new confidence semantics; regression: old tickets still classify.

## Phase 4 — Human feedback loop

**Goal:** UI thumbs → DB → exportable retraining data.

- **`src/api/routes.py`** — `POST /tickets/{id}/feedback` (`{"feedback":"up"|"down","correct_category":Optional[str]}`); `GET /feedback/export` → CSV.
- **`src/db/schemas.py`** — `FeedbackRequest`/`FeedbackResponse`.
- **`src/services/ticket_service.py`** — `record_feedback()` (validate completed ticket; set `feedback`; if a gold category is given, stash `predicted_text = text\n[gold:...]` as a golden-label seed); analytics gains `feedback_up`/`feedback_down`.
- **`streamlit_app.py`** — 👍/👎 buttons under each completed result; 👎 reveals a category selectbox; `session_state` guard against double-submit.

**Verify:** submit ticket → thumbs up/down → shows in analytics; export CSV contains the feedback rows.

## Phase 5 — Explainability panel

**Goal:** "Why this category?" — top contributing words, no extra dependency (linear coefficients).

- **New `src/services/explainability.py`** — take the ticket's TF-IDF row × classifier `coef_` for the predicted class; keep present tokens; top-K by `weight * x` magnitude → `[{word, weight, direction}]`.
  - With the ensemble, unwrap the calibrated SVC member (`named_steps["svc"]`) for `coef_`; document which member is used.
- **`src/api/routes.py`** — `GET /tickets/{id}/explain`; **`src/db/schemas.py`** — `ExplanationResponse`.
- **`streamlit_app.py`** — collapsible "Why this classification?" panel (pushed-toward / pushed-away word bars), cached per ticket id.

**Verify:** classify a sample → panel shows sensible words for e.g. "Billing Issue".

## Phase 6 — Pluggable queue (local + SQS)

**Goal:** the decoupled async architecture becomes real and demoable without AWS. Producer/consumer, with an in-process fallback.

- **New `src/queue/`** — `base.py` (`TicketQueue` protocol), `local_queue.py` (SQLite `queue_tasks` table: enqueue / claim-oldest-pending / ack), `sqs_queue.py` (adapter over the existing `aws/sqs_client.send_message` — fixes the dead path), `__init__.py` `get_queue()` factory on `QUEUE_BACKEND` (`sqs` | `local` | `inprocess`, default `inprocess`).
- **`src/services/ticket_service.py`** — `create_ticket`: after insert, `try: queue.enqueue(id) except: process_ticket_background(id)` (today's behavior as the fallback). No double-processing: the API never processes when a real backend is set.
- **`worker.py`** — generic loop (dequeue → process → ack) over any backend.
- **`docker-compose.yml`** — worker/API get `QUEUE_BACKEND=local`; `.env.example` gets `QUEUE_BACKEND=inprocess`.
- **Tests** — `tests/test_queue.py`: local enqueue→dequeue→ack roundtrip; inline fallback completes a ticket when the queue is unavailable.

**Verify:** `QUEUE_BACKEND=local` + worker → ticket completes via the worker (check logs); `QUEUE_BACKEND=inprocess` → completes inline as today.

## Phase 7 — CI/CD hardening

**Goal:** prod-grade CI without breaking the existing deploy.

- **`.github/workflows/deploy.yml`** — test job adds `ruff check src/ tests/ streamlit_app.py worker.py` and `alembic upgrade head` before pytest; new `security` job (`gitleaks` + `pip-audit`), `if: always()`.
- **New `pyproject.toml`** — ruff config + `[project]` metadata. First-pass ruff fix commit for existing offenders (`src/exception.py` unused import; long CSS strings get targeted `noqa`).
- **`docker-compose.prod.yml`** — add the `mlflow` service.
- **Risk flagged:** `pip-audit` may report known vulns in pinned old deps — whitelist via ignore file or accept a documented report (honesty beats silent fixes).

**Verify:** push a branch → GH Actions runs ruff + alembic + pytest; security job passes/ignores as configured.

## Phase 8 — Repo hygiene + docs

**Goal:** README/license/metadata that read like a real open-source project.

- **Delete** the 0-byte `notebook/2.ModelTraining.ipynb`.
- **`README.md`** — rewrite: correct entry point (`uvicorn src.app:app`), real CLI (`python -m src.pipeline.train_pipeline`), real metrics (baseline 0.54/0.81 → new ensemble F1), local-only quickstart (SQLite + `TESTING=true`), MLflow/Alembic/queue/feedback/explainability sections, FAQ on `TESTING=true`.
- **`LICENSE`** — MIT.
- **`setup.py`** — read `requirements.prod.txt` (fixes the UTF-16 bug path).
- **`.env.example`** — add `QUEUE_BACKEND`, `MLFLOW_TRACKING_URI`, `MLFLOW_MODEL_VERSION`, `CHECK_DB_SCHEMA`.
- **Update `CLAUDE.md`** to reflect the new architecture (queue backends, migrations, MLflow).

**Verify:** fresh clone → follow README → app boots, trains, and classifies with zero AWS.

---

## Cross-cutting decisions

- **AWS stays intact but optional.** SES/CloudWatch already `try/except`; they degrade silently offline. Phase 0 makes S3 model downloads non-fatal when unconfigured. The whole demo path is `SQLite + TESTING=true + local MLflow + local queue`.
- **Keep fragile pins** (streamlit 1.39.0, pandas 2.2.3, uvicorn 0.32.1, pydantic 2.7.0, fastapi 0.115.6, sqlalchemy 2.0.23, scipy 1.17.1). Git history shows a painful conflict chain — don't bump casually. **New deps added: `alembic`, `mlflow`, `optuna`, `imbalanced-learn` (explicit), `python-json-logger`. CI-only: `ruff`, `gitleaks`, `pip-audit`.**
- **MLflow version** is the top break risk → spike first (Phase 2).

## Verification loop (per phase)

1. `pip install -r requirements.prod.txt`
2. `TESTING=true pytest tests/ -v` → green
3. `uvicorn src.app:app` → `GET /health` shows db+models checks
4. `python -m src.pipeline.train_pipeline` → new artifacts in `models/`, MLflow run recorded
5. `streamlit run streamlit_app.py` → classify a sample; see model_version + explanation + feedback; export feedback CSV

## Top risks

- **MLflow/scipy pin clash** (Phase 2 spike catches early).
- **CalibratedClassifierCV memory** on 20k×25k sparse matrix (cap trials, `n_jobs=2`).
- **Alembic batch-mode on SQLite** (`render_as_batch=True`); test 0001→0002 on SQLite and Postgres.
- **ruff** flags existing code → one first-pass fix commit in the same PR as Phase 7.
- **Committed live keys** — rotated in Phase 0; full history scrub only possible on a private remote (deferred, owner's call).
