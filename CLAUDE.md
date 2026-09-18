# CLAUDE.md — Breast Cancer Diagnosis Classifier

## Context
This is YESH's second internship-resume production app (first: MNIST Digit
Classifier). Rebuilding a classroom Dense-NN notebook (96.49% test acc on
sklearn's breast cancer dataset) into a deployed, production-style app.

## Locked Decisions — do not re-litigate mid-build
- Dataset: `sklearn.datasets.load_breast_cancer()`, 30 features, label
  convention 0=malignant/1=benign (see schema.md).
- Model architecture: Dense(32)->Dropout(0.3)->Dense(16)->Dropout(0.2)->Dense(2,softmax),
  EarlyStopping, stratified 80/10/10 split (see architecture.md).
- Stack: FastAPI backend (Docker->Render) + vanilla HTML/JS frontend (Vercel),
  matching the MNIST project pattern exactly.
- Metric priority: malignant-class recall matters as much as raw accuracy.

## Working Style
- Bundle multiple steps/instructions per message — don't ask one question at a time.
- For Cursor build tasks: larger chunks of work per message, minimal
  back-and-forth, delegate execution rather than doing minimum-necessary work.
- Commit + push with conventional commit messages at every safe checkpoint,
  proactively, without being asked.
- Casual/direct communication, no fluff, honest tradeoffs over hype.

## Known Pitfall (carried from MNIST build)
- Exclude `venv/` from the Docker build context from the start — MNIST hit a
  slow, oversized build (1.75GB context) from forgetting this.

## Anchor Protocol
End sessions with an Anchor Checkpoint (decisions locked, files/state,
next step). Resume future sessions by pasting that anchor rather than
re-explaining the project.
