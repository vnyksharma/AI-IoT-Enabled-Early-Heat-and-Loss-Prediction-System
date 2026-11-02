
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "datasets" / "raw"
    / "Three-Phase-Induction-Motor-Design-Data-main"
    / "EMD Data Sheet - Original-200X97 Samples.csv"
)

MODEL_FILE = ROOT / "models" / "loss_estimator.joblib"

TARGET = "At Full Load : Losses(W)"
RATING = "Full Load Output(kW)"

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


# Load data
df = pd.read_csv(DATA_FILE)

X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(df[TARGET], errors="coerce")
rating = pd.to_numeric(df[RATING], errors="coerce")

valid = y.notna() & rating.notna() & (y > 0)

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)
rating = rating.loc[valid].reset_index(drop=True)

# Group nearby ratings together.
# Each fold will hold out a complete rating range.
groups = pd.qcut(
    rating,
    q=5,
    labels=False,
    duplicates="drop"
)

splitter = GroupKFold(n_splits=5)

model_template = joblib.load(MODEL_FILE)

baseline_template = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DummyRegressor(strategy="median"))
])

all_actual = []
all_predictions = []
all_baseline = []

print("\nUNSEEN RATING-RANGE VALIDATION")
print("Total original rows:", len(X))
print("Number of folds:", splitter.get_n_splits())

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=groups), start=1
):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = clone(model_template)
    baseline = clone(baseline_template)

    model.fit(X_train, y_train)
    baseline.fit(X_train, y_train)

    predictions = model.predict(X_test)
    baseline_predictions = baseline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\nFOLD {fold}")
    print(
        "Test rating range:",
        round(rating.iloc[test_idx].min(), 2),
        "to",
        round(rating.iloc[test_idx].max(), 2),
        "kW"
    )
    print("Test rows:", len(test_idx))
    print("Model MAE:", round(mae, 2), "W")
    print("Model RMSE:", round(rmse, 2), "W")
    print(
        "Baseline MAE:",
        round(mean_absolute_error(y_test, baseline_predictions), 2),
        "W"
    )

    all_actual.extend(y_test.to_numpy())
    all_predictions.extend(predictions)
    all_baseline.extend(baseline_predictions)


# Overall results
all_actual = np.array(all_actual)
all_predictions = np.array(all_predictions)
all_baseline = np.array(all_baseline)

model_mae = mean_absolute_error(all_actual, all_predictions)
baseline_mae = mean_absolute_error(all_actual, all_baseline)
model_rmse = np.sqrt(mean_squared_error(all_actual, all_predictions))
model_r2 = r2_score(all_actual, all_predictions)

print("\nOVERALL UNSEEN-RANGE RESULTS")
print("Model MAE:", round(model_mae, 2), "W")
print("Model RMSE:", round(model_rmse, 2), "W")
print("Model R2:", round(model_r2, 6))
print("Baseline MAE:", round(baseline_mae, 2), "W")

if baseline_mae > 0:
    improvement = (1 - model_mae / baseline_mae) * 100
    print("Improvement over baseline:", round(improvement, 2), "%")

print("\nVALIDATION FINISHED")
print("No dashboard or historical database was changed.")
