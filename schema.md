# schema.md — Breast Cancer Diagnosis Classifier

## Input Features (30 total, all floats)
Grouped into 3 statistics x 10 base measurements, matching sklearn's
`load_breast_cancer().feature_names` order exactly:

Base measurements (x10): radius, texture, perimeter, area, smoothness,
compactness, concavity, concave_points, symmetry, fractal_dimension

Groups (x3): `mean_*`, `se_*` (standard error), `worst_*`

Full ordered list (must match training column order exactly):
```
mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
mean_fractal_dimension,
se_radius, se_texture, se_perimeter, se_area, se_smoothness,
se_compactness, se_concavity, se_concave_points, se_symmetry,
se_fractal_dimension,
worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness,
worst_compactness, worst_concavity, worst_concave_points, worst_symmetry,
worst_fractal_dimension
```

## API Request Schema — `POST /predict`
```json
{
  "mean_radius": 11.76,
  "mean_texture": 21.6,
  "...": "... all 30 fields, floats"
}
```
Pydantic model `PredictRequest` with all 30 fields required, typed `float`,
field names as above (snake_case, matches training-time column renaming from
sklearn's default names like `mean concave points`).

## API Response Schema
```json
{
  "prediction": "Benign",
  "confidence": 0.9006,
  "probabilities": {
    "malignant": 0.0775,
    "benign": 0.9006
  }
}
```

## Label Convention
- sklearn's `load_breast_cancer` encodes: `0 = malignant`, `1 = benign`
  (already verified against the notebook's output — do not flip this).
- Output labels in the API/frontend must be human-readable strings
  ("Malignant"/"Benign"), never raw 0/1.

## Model Artifact
- Saved as `breast_cancer_nn.keras`
- Companion `scaler.pkl` (StandardScaler, fit on training data only) — MUST be
  shipped and applied identically at inference time, same as the notebook did.
- Feature order in the scaler/model must exactly match the schema list above.

## Sample Cases (for frontend "Load Sample Case")
- Take one confirmed-malignant and one confirmed-benign row directly from the
  dataset (not fabricated) so demo predictions are trustworthy.
