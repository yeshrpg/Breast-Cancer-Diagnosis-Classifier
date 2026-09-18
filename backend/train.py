"""
train.py — Breast Cancer Diagnosis Classifier
Uses stratified 5-fold CV to get a trustworthy performance estimate on this
small (569-row) dataset, then trains the final deployed model on a full
85/15 train/test split. Reports mean +/- std across folds instead of a
single noisy split.
"""

import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
import joblib

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)

FEATURE_NAMES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
    "mean_fractal_dimension",
    "se_radius", "se_texture", "se_perimeter", "se_area", "se_smoothness",
    "se_compactness", "se_concavity", "se_concave_points", "se_symmetry",
    "se_fractal_dimension",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area", "worst_smoothness",
    "worst_compactness", "worst_concavity", "worst_concave_points", "worst_symmetry",
    "worst_fractal_dimension",
]

data = sklearn.datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns=FEATURE_NAMES)
df["label"] = data.target  # 0 = malignant, 1 = benign

X_full = df[FEATURE_NAMES].values
y_full = df["label"].values

print("Dataset shape:", df.shape)
print("Class balance:\n", df["label"].value_counts())


def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(30,)),
        keras.layers.Dense(32, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3)),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# Class weights (mild imbalance: 63% benign / 37% malignant)
# ---------------------------------------------------------------------------
class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_full), y=y_full)
class_weight = {i: w for i, w in enumerate(class_weights_arr)}
print("Class weights:", class_weight)

# ---------------------------------------------------------------------------
# Step 1: Stratified 5-fold CV -> trustworthy metric (report mean +/- std)
# ---------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_acc, fold_recall, fold_precision = [], [], []

print("\n=== 5-Fold Cross-Validation ===")
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), start=1):
    X_tr, X_va = X_full[train_idx], X_full[val_idx]
    y_tr, y_va = y_full[train_idx], y_full[val_idx]

    scaler_cv = StandardScaler().fit(X_tr)
    X_tr_std, X_va_std = scaler_cv.transform(X_tr), scaler_cv.transform(X_va)

    model_cv = build_model()
    early_stop_cv = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    model_cv.fit(
        X_tr_std, y_tr,
        validation_data=(X_va_std, y_va),
        epochs=200, batch_size=16,
        class_weight=class_weight,
        callbacks=[early_stop_cv],
        verbose=0,
    )

    y_pred = np.argmax(model_cv.predict(X_va_std, verbose=0), axis=1)
    acc = accuracy_score(y_va, y_pred)
    rec = recall_score(y_va, y_pred, pos_label=0)  # malignant recall
    prec = precision_score(y_va, y_pred, pos_label=0)

    fold_acc.append(acc)
    fold_recall.append(rec)
    fold_precision.append(prec)
    print(f"Fold {fold_idx}: accuracy={acc:.4f}  malignant_recall={rec:.4f}  malignant_precision={prec:.4f}")

print(f"\nCV accuracy:            {np.mean(fold_acc):.4f} +/- {np.std(fold_acc):.4f}")
print(f"CV malignant recall:    {np.mean(fold_recall):.4f} +/- {np.std(fold_recall):.4f}")
print(f"CV malignant precision: {np.mean(fold_precision):.4f} +/- {np.std(fold_precision):.4f}")
print("^ This is the number to put in the README, not a single split's accuracy.")

# ---------------------------------------------------------------------------
# Step 2: Train the FINAL deployed model on an 85/15 train/test split
# (more training data than before -> more stable single model)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.15, stratify=y_full, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

model = build_model()
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

history = model.fit(
    X_train_std, y_train,
    validation_split=0.15,  # carved from the 85% train portion
    epochs=200,
    batch_size=16,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=2,
)

test_loss, test_acc = model.evaluate(X_test_std, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test_std, verbose=0), axis=1)

print(f"\n=== Final Model — Held-out Test Set ===")
print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
print("\nConfusion matrix (rows=true, cols=pred) [0=malignant, 1=benign]:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))
print(f"Malignant-class recall: {recall_score(y_test, y_pred, pos_label=0):.4f}")

# ---------------------------------------------------------------------------
# Save model + scaler
# ---------------------------------------------------------------------------
model.save("breast_cancer_nn.keras")
joblib.dump(scaler, "scaler.pkl")
print("\nSaved: breast_cancer_nn.keras, scaler.pkl")

# Sample cases for frontend
malignant_idx = np.where(y_full == 0)[0][0]
benign_idx = np.where(y_full == 1)[0][0]
print("\nSample MALIGNANT case (raw, unscaled):")
print(dict(zip(FEATURE_NAMES, X_full[malignant_idx].tolist())))
print("\nSample BENIGN case (raw, unscaled):")
print(dict(zip(FEATURE_NAMES, X_full[benign_idx].tolist())))
