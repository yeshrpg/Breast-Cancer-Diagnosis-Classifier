# architecture.md — Breast Cancer Diagnosis Classifier

## Stack
- **Model**: Keras/TensorFlow Dense NN (upgraded from notebook baseline)
- **Backend**: FastAPI (Python), Dockerized
- **Frontend**: Vanilla HTML/CSS/JS, single-page form
- **Backend deploy**: Render (Docker web service) — same as MNIST project
- **Frontend deploy**: Vercel — same as MNIST project

## Repo Layout
```
breast-cancer-classifier/
├── PRD.md
├── schema.md
├── architecture.md
├── tasks.md
├── CLAUDE.md
├── backend/
│   ├── train.py            # trains + saves model + scaler
│   ├── model.py            # loads model/scaler, exposes predict()
│   ├── schemas.py           # Pydantic request/response models
│   ├── main.py              # FastAPI app, /predict, /health
│   ├── breast_cancer_nn.keras
│   ├── scaler.pkl
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .gitignore
└── frontend/
    ├── index.html
    └── script.js
```

## Model Architecture (upgrade over notebook baseline)
Notebook baseline: `Flatten -> Dense(20, relu) -> Dense(2, sigmoid)`,
sparse_categorical_crossentropy, 10 epochs, no regularization, 96.49% test acc.

Upgraded design (locked, do not re-decide mid-build):
- Input: 30 standardized features (StandardScaler, fit on train only)
- `Dense(32, relu)` -> `Dropout(0.3)` -> `Dense(16, relu)` -> `Dropout(0.2)`
  -> `Dense(2, softmax)`
- Loss: `sparse_categorical_crossentropy`, optimizer: `adam`
- `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`
- Train/val/test split: 80/10/10 (stratified, since classes are imbalanced
  ~63/37 malignant/benign)
- Report accuracy AND recall/precision/confusion matrix per class in the
  training output — recall on malignant class is the metric that matters most.

## Backend Flow
1. `main.py` loads `breast_cancer_nn.keras` + `scaler.pkl` once at startup.
2. `POST /predict` -> Pydantic validates 30 floats -> order fields per schema.md
   -> scale with `scaler.transform` -> `model.predict` -> argmax + softmax probs
   -> map 0/1 to Malignant/Benign -> return JSON.
3. `GET /health` -> `{"status": "ok"}`.
4. CORS enabled for the deployed frontend origin (and localhost for dev).

## Frontend Flow
1. Single page, 3 collapsible sections (Mean / SE / Worst), 10 numeric inputs each.
2. "Load Sample Case" buttons (malignant example, benign example) autofill all 30 fields.
3. "Predict" button -> `fetch(POST /predict)` -> render result card:
   predicted class, confidence %, small note that this isn't medical advice.
4. No frameworks — matches MNIST project's plain JS approach for consistency.

## Deployment
- Backend: same Docker+Render pattern as MNIST (Dockerfile, `.gitignore` excludes
  `venv/`, exposed port, health check endpoint for Render).
- Frontend: static site on Vercel, `script.js` points at deployed backend URL
  via a config constant (update after backend is live).

## Docker Notes (carried over lesson from MNIST build)
- Exclude `venv/` from Docker build context via `.gitignore`/`.dockerignore`
  to avoid the slow, oversized build MNIST hit.
