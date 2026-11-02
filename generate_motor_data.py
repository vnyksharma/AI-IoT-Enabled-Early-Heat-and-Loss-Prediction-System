
import numpy as np
import pandas as pd
from pathlib import Path

# Reproducible synthetic data
rng = np.random.default_rng(42)

OUTPUT = Path("datasets/synthetic")
OUTPUT.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_SCENARIO = 3000
INTERVAL_SECONDS = 3

scenarios = {
    "normal": {
        "start_temp": 29.5,
        "heating_rate": 0.0005,
        "vibration": 0.006,
        "heat_start": 999999,
    },
    "gradual_heating": {
        "start_temp": 29.5,
        "heating_rate": 0.018,
        "vibration": 0.009,
        "heat_start": 500,
    },
    "rapid_heating": {
        "start_temp": 29.5,
        "heating_rate": 0.045,
        "vibration": 0.012,
        "heat_start": 250,
    },
    "abnormal_operation": {
        "start_temp": 29.5,
        "heating_rate": 0.012,
        "vibration": 0.035,
        "heat_start": 500,
    },
}

all_data = []

for scenario, settings in scenarios.items():

    for run in range(10):

        temperature = settings["start_temp"]
        run_id = f"{scenario}_{run + 1}"

        for step in range(SAMPLES_PER_SCENARIO):

            if step >= settings["heat_start"]:
                temperature += settings["heating_rate"]

            temperature += rng.normal(0, 0.025)

            vibration = (
                settings["vibration"]
                + rng.normal(0, settings["vibration"] * 0.12)
            )

            vibration = max(0.0001, vibration)

            all_data.append({
                "run_id": run_id,
                "scenario": scenario,
                "time_seconds": step * INTERVAL_SECONDS,
                "temperature_c": round(temperature, 4),
                "vibration_rms_g": round(vibration, 6),
            })

df = pd.DataFrame(all_data)

# Synthetic labels for prototype development only
df["heat_label"] = (
    df["scenario"].isin([
        "gradual_heating",
        "rapid_heating"
    ])
).astype(int)

df["abnormal_operation_label"] = (
    df["scenario"] == "abnormal_operation"
).astype(int)

output_file = OUTPUT / "synthetic_motor_readings.csv"
df.to_csv(output_file, index=False)

print("SYNTHETIC DATASET CREATED")
print("-------------------------")
print("Rows:", len(df))
print("Scenarios:", df["scenario"].nunique())
print("\nScenario distribution:")
print(df["scenario"].value_counts())
print("\nSaved to:", output_file)
print("\nSynthetic data only. Not real motor measurements.")
