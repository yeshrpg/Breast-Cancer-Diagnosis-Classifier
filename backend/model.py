# model.py
import numpy as np
import joblib
from tensorflow import keras

MODEL_PATH = "breast_cancer_nn.keras"
SCALER_PATH = "scaler.pkl"

_model = keras.models.load_model(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)

FEATURE_ORDER = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
    "radius_error", "texture_error", "perimeter_error", "area_error",
    "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
]

def predict(features: dict) -> dict:
    x = np.array([[features[f] for f in FEATURE_ORDER]])
    x_scaled = _scaler.transform(x)
    probs = _model.predict(x_scaled, verbose=0)[0]  # [P(malignant), P(benign)]
    label = int(np.argmax(probs))
    return {
        "prediction": "malignant" if label == 0 else "benign",
        "label": label,
        "confidence": float(probs[label]),
        "probabilities": {
            "malignant": float(probs[0]),
            "benign": float(probs[1]),
        },
    }