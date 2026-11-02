
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "datasets" / "raw"
    / "Three-Phase-Induction-Motor-Design-Data-main"
    / "EMD Data Sheet - Original-200X97 Samples.csv"
)

MODEL_FILE = ROOT / "models" / "loss_estimator.joblib"

TARGET = "At Full Load : Losses(W)"

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

df = pd.read_csv(DATA_FILE)
X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")

valid = y.notna() & (y > 0)
X = X.loc[valid]
y = y.loc[valid]

model = joblib.load(MODEL_FILE)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
predictions = cross_val_predict(model, X, y, cv=cv)

errors = predictions - y.to_numpy()

print("\nTARGET DISTRIBUTION")
print("Minimum:", round(y.min(), 2), "W")
print("Median:", round(y.median(), 2), "W")
print("Maximum:", round(y.max(), 2), "W")

print("\nCROSS-VALIDATION CHECK")
print("MAE:", round(mean_absolute_error(y, predictions), 2), "W")
print("R2:", round(r2_score(y, predictions), 6))
print("Mean signed error:", round(errors.mean(), 2), "W")
print("Maximum absolute error:", round(np.abs(errors).max(), 2), "W")

print("\nWORST 10 PREDICTIONS")
results = pd.DataFrame({
    "Actual_W": y.to_numpy(),
    "Predicted_W": predictions,
    "Absolute_Error_W": np.abs(errors),
})
print(
    results.nlargest(10, "Absolute_Error_W")
    .round(2)
    .to_string(index=False)
)

print("\nTOP 15 FEATURE IMPORTANCES")
fitted_model = model.named_steps["model"]

importance = pd.Series(
    fitted_model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

print(importance.head(15).round(4).to_string())

print("\nVALIDATION REPORT COMPLETE")
