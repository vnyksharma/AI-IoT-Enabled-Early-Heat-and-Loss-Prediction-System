
from pathlib import Path
import json
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "datasets" / "raw"
    / "Three-Phase-Induction-Motor-Design-Data-main"
    / "EMD Data Sheet - Original-200X97 Samples.csv"
)

MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = MODEL_DIR / "loss_reference_1hp.json"

# --------------------------------------------------
# 2. Verified motor nameplate specifications
# --------------------------------------------------

MOTOR = {
    "manufacturer": "CG",
    "rated_power_kw": 0.75,
    "rated_power_hp": 1.0,
    "voltage_v": 415,
    "frequency_hz": 50,
    "phases": 3,
    "rated_speed_rpm": 1390,
    "rated_current_a": 1.70,
    "efficiency_percent": 79.60,
    "power_factor": 0.77,
}

# --------------------------------------------------
# 3. Load original dataset
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

rating_column = "Full Load Output(kW)"
loss_column = "At Full Load : Losses(W)"

# Select only nearby low-power motor designs.
reference = df[
    (df[rating_column] >= 0.5)
    & (df[rating_column] <= 1.0)
][[rating_column, loss_column]].copy()

reference = reference.sort_values(rating_column)

if len(reference) < 2:
    raise ValueError("At least two reference designs are required.")

ratings = reference[rating_column].to_numpy(dtype=float)
losses = reference[loss_column].to_numpy(dtype=float)

target_power = MOTOR["rated_power_kw"]

# --------------------------------------------------
# 4. Estimate using linear interpolation
# --------------------------------------------------

if target_power < ratings.min() or target_power > ratings.max():
    raise ValueError(
        "Motor rating is outside the reference range. "
        "Extrapolation is not allowed."
    )

estimated_loss = float(
    np.interp(target_power, ratings, losses)
)

# --------------------------------------------------
# 5. Save reference model
# --------------------------------------------------

model_data = {
    "model_name": "MotorAI 1HP Design Loss Reference",
    "method": "Linear interpolation",
    "motor": MOTOR,
    "reference_dataset": DATA_FILE.name,
    "reference_points": [
        {
            "rated_output_kw": float(row[rating_column]),
            "full_load_losses_w": float(row[loss_column]),
        }
        for _, row in reference.iterrows()
    ],
    "estimated_full_load_losses_w": round(estimated_loss, 2),
    "reference_rating_range_kw": [
        float(ratings.min()),
        float(ratings.max()),
    ],
    "limitations": [
        "Design-based reference, not measured operating losses.",
        "Dataset reference voltage is 400 V; motor nameplate is 415 V.",
        "Dataset efficiency and power factor differ from the actual motor.",
        "Estimate is not independently validated against this motor.",
    ],
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(model_data, file, indent=4)

# --------------------------------------------------
# 6. Display results
# --------------------------------------------------

print("\nMOTORAI 1 HP LOSS REFERENCE")
print("--------------------------")

print("Motor:", MOTOR["manufacturer"])
print("Rated output:", MOTOR["rated_power_kw"], "kW")
print("Voltage:", MOTOR["voltage_v"], "V")
print("Frequency:", MOTOR["frequency_hz"], "Hz")

print("\nREFERENCE DATA")
print(reference.to_string(index=False))

print("\nESTIMATED FULL-LOAD DESIGN LOSSES")
print(round(estimated_loss, 2), "W")

print("\nSaved reference:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("This is a preliminary design-based estimate.")
print("It is not a measurement of actual motor losses.")
