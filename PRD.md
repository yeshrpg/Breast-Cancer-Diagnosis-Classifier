# PRD.md — Breast Cancer Diagnosis Classifier

## 1. Overview
A resume-ready, production-style web app that predicts whether a breast tumor
is **Malignant** or **Benign** from 30 numeric diagnostic features (from the
Wisconsin Diagnostic Breast Cancer dataset via `sklearn.datasets.load_breast_cancer`).

Rebuilt from a classroom notebook (simple Dense NN, 96.49% test accuracy) into
a real deployed app: FastAPI backend + trained/tuned Keras model, vanilla
HTML/JS frontend form, Dockerized, deployed live.

## 2. Goals
- Take the notebook's Dense NN baseline and turn it into a properly validated,
  tuned model (target: **>=97.5% test accuracy**, with attention to recall on
  the malignant class — false negatives are the costly error here).
- Ship a real deployed product: backend API + frontend form, both live on the
  internet, matching the MNIST project's production bar.
- Second internship-resume piece, same tier of polish as MNIST project.

## 3. Non-Goals
- No user accounts, no database, no history of past predictions.
- Not a medical device — no real diagnostic claims; include a visible disclaimer.
- Not retraining on user-submitted data.

## 4. Users & Use Case
Single-user demo/portfolio tool. A visitor (recruiter, reviewer) enters the
30 measurements (or loads a sample case) and instantly sees a prediction with
confidence score.

## 5. Functional Requirements
- FR1: Backend exposes `POST /predict` accepting 30 named numeric features,
  returns predicted class (Malignant/Benign) + probability for both classes.
- FR2: Backend exposes `GET /health` for uptime checks.
- FR3: Frontend presents the 30 inputs grouped into 3 logical sections
  (Mean / Standard Error / Worst — matching the dataset's own grouping of
  10 base measurements x 3 statistics each).
- FR4: Frontend has a "Load Sample Case" button (at least 2 sample cases: one
  malignant, one benign) so reviewers can test without manually typing 30 numbers.
- FR5: Frontend displays prediction, confidence %, and a clear non-diagnostic disclaimer.
- FR6: Input validation both client-side (numeric, required) and server-side (Pydantic).

## 6. Success Metrics
- Test accuracy >= 97.5%, recall on malignant class prioritized in reporting.
- Live backend + frontend, end-to-end prediction working in production.
- Clean README with architecture diagram, metrics, and live links.

## 7. Constraints
- Same working pattern as MNIST project: spec-first, Tier-1 files locked before code.
- Small dataset (569 rows) — must guard against overfitting (dropout/early stopping/
  regularization), not just add layers blindly.
- Commit + push at every safe checkpoint with conventional commit messages.
