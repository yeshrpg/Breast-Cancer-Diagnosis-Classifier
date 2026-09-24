# 🩺 Breast Cancer Diagnosis Classifier

A full-stack, production-deployed neural network system that classifies breast tumor samples as **malignant** or **benign** from 30 diagnostic features — trained, served, and deployed end-to-end.

**🔗 Live App:** [breast-cancer-diagnosis-classifier-omega.vercel.app](https://breast-cancer-diagnosis-classifier-omega.vercel.app/)

**⚙️ API:** [breast-cancer-diagnosis-classifier-csrm.onrender.com](https://breast-cancer-diagnosis-classifier-csrm.onrender.com)

---

## ✨ What makes this different

Most "ML classifier" portfolio projects stop at a Jupyter notebook with a printed accuracy score. This one doesn't:

- **Actually deployed, not just trained** — the model is wrapped in a real API and served publicly, not left inside a `.ipynb`.
- **Validated the honest way** — stratified 5-fold cross-validation reported *alongside* a final holdout model, instead of quoting a single lucky train/test split.
- **Recall-first, not accuracy-first** — the model is tuned to prioritize catching malignant cases (minimizing false negatives), which is the metric that actually matters in a diagnostic context, not just overall accuracy.
- **Containerized like a real service** — Dockerized backend, deployed the same way a production ML service would be, not just `python app.py` on a laptop.
- **Full request/response contract** — typed request/response schemas (Pydantic), input validation, and a clean `/health` + `/predict` API surface, not a single hacked-together endpoint.

## 🧠 Model

| | |
|---|---|
| **Dataset** | `sklearn.datasets.load_breast_cancer()` — 30 features, 569 samples |
| **Architecture** | `Dense(32, relu, L2=1e-3) → Dropout(0.2) → Dense(16, relu, L2=1e-3) → Dropout(0.1) → Dense(2, softmax)` |
| **Training** | Adam optimizer, class-weight balanced (handles class imbalance) |
| **Validation** | Stratified 5-fold cross-validation for the trustworthy metric + a final model trained on an 85/15 split |
| **CV Accuracy** | **97.72% ± 0.89%** |
| **CV Malignant Recall** | **96.25% ± 3.16%** |
| **CV Malignant Precision** | **97.69% ± 2.03%** |

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────────┐         ┌────────────────────┐
│  Vercel          │  POST   │  Render (Docker)      │         │  TensorFlow / Keras │
│  Vanilla HTML/JS │ ──────> │  FastAPI + Uvicorn     │ ──────> │  Model + Scaler      │
│  Frontend        │ <────── │  /predict  /health     │ <────── │  (.keras + .pkl)     │
└─────────────────┘  JSON   └──────────────────────┘         └────────────────────┘
```

**Stack**
- **Model:** TensorFlow / Keras
- **Backend:** FastAPI, Pydantic, served via Uvicorn, containerized with Docker
- **Hosting:** Render (API), Vercel (static frontend)
- **Frontend:** Vanilla HTML/CSS/JS — no framework overhead

## 🚀 Features

- Manual entry of all 30 diagnostic features, or one-click **sample case loading** (real malignant/benign examples from the dataset)
- Real-time prediction with confidence score and per-class probability breakdown
- Input validation on both frontend and backend
- CORS-enabled public API, independently usable/testable via `curl` or Postman

## 🧪 Running Locally

**Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend**
```bash
cd frontend
# just open index.html in a browser — no build step needed
```

**Docker (backend)**
```bash
cd backend
docker build -t bc-classifier-api .
docker run -p 8000:8000 bc-classifier-api
```

## 📡 API Reference

**`GET /health`**
```json
{ "status": "ok" }
```

**`POST /predict`**
Body: JSON object with all 30 feature keys (see `backend/schemas.py`)

Response:
```json
{
  "prediction": "malignant",
  "label": 0,
  "confidence": 1.0,
  "probabilities": { "malignant": 1.0, "benign": 0.0 }
}
```

## ⚠️ Disclaimer

This is an educational/portfolio project, **not a certified medical diagnostic tool**. Do not use it for actual clinical decision-making.

## 📌 Next Steps

- [ ] Add authentication/rate-limiting to the public API before wider sharing
- [ ] Add a confusion matrix / ROC curve visualization page to the frontend
- [ ] Add SHAP or feature-importance explainability to show *why* a prediction was made
- [ ] Write unit tests for `model.py` and `main.py` (pytest + FastAPI TestClient)
- [ ] Add CI (GitHub Actions) to run tests on push before deploy
- [ ] Upgrade free-tier Render plan (or add a keep-alive ping) to eliminate cold-start delay
- [ ] Optional: swap CORS `allow_origins=["*"]` for the exact Vercel domain once stable
