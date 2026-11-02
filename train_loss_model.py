
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT
    / "datasets"
    / "raw"
    / "Three-Phase-Induction-Motor-Design-Data-main"
    / "EMD Data Sheet - Original-200X97 Samples.csv"
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "datasets" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load original dataset
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

TARGET = "At Full Load : Losses(W)"

# Only design/rated parameters.
# Loss columns and direct target-related columns are excluded.
FEATURES = [
    "Full Load Output(kW)",
    "Line Voltage(V)",
    "Frequency(Hz)",
    "Phases",
    "Power Factor",
    "Number of Poles",
    "Specific Magnetic Loading(Wb/m2)",
    "Specific Electric Loading(A/m)",
    "Stator Bore(mm)",
    "Gross Iron Length(mm)",
    "Net Iron Length(mm)",
    "Pole Pitch(mm)",
    "Turns Per Phase",
    "Number of Slots",
    "Coil Span",
    "Winding Factor",
    "Current density(A/mm2)",
    "Conductor:area(mm2)",
    "Length of Mean Turn(mm)",
    "Depth of Stator Core(mm)",
    "Outer Diameter Stator Lamination(mm)",
    "Length of Air Gap(mm)",
    "Diameter of Rotor(mm)",
    "Rotor Bar:Area(mm2)",
    "Rotor Bar:Length(mm)",
    "Depth of Rotor Core(mm)",
]

missing = [column for column in FEATURES + [TARGET] if column not in df.columns]

if missing:
    raise ValueError(f"Missing dataset columns: {missing}")

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")

valid = y.notna() & (y > 0)
X = X.loc[valid].copy()
y = y.loc[valid].copy()

if len(X) < 20:
    raise ValueError("Not enough valid rows to train and evaluate the model.")

print("\nDATASET")
print("Valid rows:", len(X))
print("Features:", len(FEATURES))
print("Target:", TARGET)


# --------------------------------------------------
# 3. Define models
# --------------------------------------------------

baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DummyRegressor(strategy="median"))
])

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    ))
])


# --------------------------------------------------
# 4. Cross-validation
# --------------------------------------------------

cv = KFold(n_splits=5, shuffle=True, random_state=42)

baseline_predictions = cross_val_predict(
    baseline, X, y, cv=cv
)

model_predictions = cross_val_predict(
    model, X, y, cv=cv
)


def calculate_metrics(actual, predicted):
    return {
        "MAE_W": float(mean_absolute_error(actual, predicted)),
        "RMSE_W": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted))
    }


baseline_metrics = calculate_metrics(y, baseline_predictions)
model_metrics = calculate_metrics(y, model_predictions)

print("\nBASELINE RESULTS")
print(json.dumps(baseline_metrics, indent=2))

print("\nEXTRA TREES RESULTS")
print(json.dumps(model_metrics, indent=2))


# --------------------------------------------------
# 5. Compare against baseline
# --------------------------------------------------

improvement = (
    1 - model_metrics["MAE_W"] / baseline_metrics["MAE_W"]
) * 100

print(f"\nMAE improvement over baseline: {improvement:.2f}%")

# Do not save a model that fails to beat the baseline.
if model_metrics["MAE_W"] >= baseline_metrics["MAE_W"]:
    print("\nModel did not beat the baseline.")
    print("No trained model was saved.")
    raise SystemExit(0)


# --------------------------------------------------
# 6. Train final model on all valid original rows
# --------------------------------------------------

model.fit(X, y)

model_path = MODEL_DIR / "loss_estimator.joblib"
joblib.dump(model, model_path)


# --------------------------------------------------
# 7. Save evaluation report
# --------------------------------------------------

report = {
    "dataset": DATA_FILE.name,
    "dataset_rows_used": int(len(X)),
    "feature_count": len(FEATURES),
    "target": TARGET,
    "validation": "5-fold shuffled cross-validation",
    "baseline": baseline_metrics,
    "model": model_metrics,
    "mae_improvement_percent": float(improvement),
    "warning": (
        "Design-based estimate only. This is not a measurement of "
        "actual operating losses. The dataset contains only 200 motor "
        "design samples; external validation is required."
    )
}

report_path = REPORT_DIR / "loss_model_evaluation.json"

with open(report_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nTRAINING COMPLETE")
print("Saved model:", model_path)
print("Saved report:", report_path)
print("\nRemember: cross-validation results are not proof of real-motor accuracy.")

