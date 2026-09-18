# tasks.md — Breast Cancer Diagnosis Classifier

## Phase 1 — Model
- [ ] T1.1: `train.py` — load `sklearn.datasets.load_breast_cancer()`, rename
      columns to schema.md's snake_case names, stratified 80/10/10 split,
      fit StandardScaler on train only, build upgraded model (architecture.md),
      train with EarlyStopping, print accuracy/precision/recall/confusion matrix,
      save `breast_cancer_nn.keras` + `scaler.pkl`.
- [ ] T1.2: Verify test accuracy >= 97.5% and malignant recall is not worse
      than the notebook baseline; iterate on architecture/epochs if not.

## Phase 2 — Backend
- [ ] T2.1: `schemas.py` — `PredictRequest` (30 float fields) + `PredictResponse`.
- [ ] T2.2: `model.py` — load model+scaler once, `predict(features: list[float]) -> dict`.
- [ ] T2.3: `main.py` — FastAPI app, `/predict`, `/health`, CORS, test via
      Swagger UI `/docs` with at least one malignant and one benign sample.

## Phase 3 — Frontend
- [ ] T3.1: `index.html` — 3 grouped sections of 10 inputs each, Predict button,
      2 "Load Sample Case" buttons, result display area, disclaimer text.
- [ ] T3.2: `script.js` — sample-case autofill, fetch to backend, render
      prediction + confidence, basic input validation.
- [ ] T3.3: End-to-end test against local backend.

## Phase 4 — Dockerize & Deploy
- [ ] T4.1: `Dockerfile` + `.gitignore`/`.dockerignore` (exclude venv up front
      this time), verify local `docker build` + `docker run` works.
- [ ] T4.2: Push repo to GitHub (conventional commits, clean history).
- [ ] T4.3: Deploy backend to Render (Docker web service).
- [ ] T4.4: Deploy frontend to Vercel, point `script.js` at live backend URL.
- [ ] T4.5: Final live integration test (both sample cases, plus a manual
      edge-case input).

## Phase 5 — Docs
- [ ] T5.1: README — problem statement, architecture, metrics table
      (accuracy/precision/recall/confusion matrix), live links, screenshots.

## Working Style (locked, same as MNIST)
- Larger work chunks per message, minimal back-and-forth, delegate build
  execution to Cursor.
- Commit + push at every safe checkpoint, conventional commit messages
  (feat:/fix:/chore:/docs:) — proactively reminded, not waited on.
