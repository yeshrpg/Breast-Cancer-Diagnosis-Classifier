// frontend/script.js
const API_URL = "https://breast-cancer-diagnosis-classifier-csrm.onrender.com/predict";

const FEATURES = [
  "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
  "mean_smoothness", "mean_compactness", "mean_concavity",
  "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
  "radius_error", "texture_error", "perimeter_error", "area_error",
  "smoothness_error", "compactness_error", "concavity_error",
  "concave_points_error", "symmetry_error", "fractal_dimension_error",
  "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
  "worst_smoothness", "worst_compactness", "worst_concavity",
  "worst_concave_points", "worst_symmetry", "worst_fractal_dimension",
];

const MALIGNANT_SAMPLE = {
  mean_radius: 17.99, mean_texture: 10.38, mean_perimeter: 122.8, mean_area: 1001.0,
  mean_smoothness: 0.1184, mean_compactness: 0.2776, mean_concavity: 0.3001,
  mean_concave_points: 0.1471, mean_symmetry: 0.2419, mean_fractal_dimension: 0.07871,
  radius_error: 1.095, texture_error: 0.9053, perimeter_error: 8.589, area_error: 153.4,
  smoothness_error: 0.006399, compactness_error: 0.04904, concavity_error: 0.05373,
  concave_points_error: 0.01587, symmetry_error: 0.03003, fractal_dimension_error: 0.006193,
  worst_radius: 25.38, worst_texture: 17.33, worst_perimeter: 184.6, worst_area: 2019.0,
  worst_smoothness: 0.1622, worst_compactness: 0.6656, worst_concavity: 0.7119,
  worst_concave_points: 0.2654, worst_symmetry: 0.4601, worst_fractal_dimension: 0.1189,
};

const BENIGN_SAMPLE = {
  mean_radius: 12.32, mean_texture: 12.39, mean_perimeter: 78.85, mean_area: 464.1,
  mean_smoothness: 0.1028, mean_compactness: 0.06981, mean_concavity: 0.03987,
  mean_concave_points: 0.037, mean_symmetry: 0.1959, mean_fractal_dimension: 0.05955,
  radius_error: 0.236, texture_error: 0.6656, perimeter_error: 1.67, area_error: 17.43,
  smoothness_error: 0.008045, compactness_error: 0.01532, concavity_error: 0.01587,
  concave_points_error: 0.009432, symmetry_error: 0.02292, fractal_dimension_error: 0.003065,
  worst_radius: 13.5, worst_texture: 15.64, worst_perimeter: 86.97, worst_area: 549.1,
  worst_smoothness: 0.1385, worst_compactness: 0.1266, worst_concavity: 0.1242,
  worst_concave_points: 0.09391, worst_symmetry: 0.2827, worst_fractal_dimension: 0.06771,
};

const grid = document.getElementById("featureGrid");
FEATURES.forEach((f) => {
  const label = document.createElement("label");
  label.textContent = f.replace(/_/g, " ");
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  input.required = true;
  input.id = f;
  label.appendChild(input);
  grid.appendChild(label);
});

function fillForm(data) {
  FEATURES.forEach((f) => {
    document.getElementById(f).value = data[f];
  });
}

document.getElementById("loadMalignant").addEventListener("click", () => fillForm(MALIGNANT_SAMPLE));
document.getElementById("loadBenign").addEventListener("click", () => fillForm(BENIGN_SAMPLE));

document.getElementById("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorDiv = document.getElementById("error");
  const resultDiv = document.getElementById("result");
  errorDiv.classList.add("hidden");
  resultDiv.classList.add("hidden");

  const payload = {};
  FEATURES.forEach((f) => {
    payload[f] = parseFloat(document.getElementById(f).value);
  });

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(JSON.stringify(err));
    }

    const data = await res.json();
    const malignantPct = (data.probabilities.malignant * 100).toFixed(2);
    const benignPct = (data.probabilities.benign * 100).toFixed(2);

    document.getElementById("resultLabel").textContent =
      data.prediction === "malignant" ? "⚠ Malignant" : "✓ Benign";
    document.getElementById("resultLabel").style.color =
      data.prediction === "malignant" ? "#c0392b" : "#27ae60";
    document.getElementById("resultConfidence").textContent =
      `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    document.getElementById("malignantBar").style.width = `${malignantPct}%`;
    document.getElementById("benignBar").style.width = `${benignPct}%`;
    document.getElementById("malignantPct").textContent = `${malignantPct}%`;
    document.getElementById("benignPct").textContent = `${benignPct}%`;

    resultDiv.classList.remove("hidden");
  } catch (err) {
    errorDiv.textContent = "Error: " + err.message;
    errorDiv.classList.remove("hidden");
  }
});