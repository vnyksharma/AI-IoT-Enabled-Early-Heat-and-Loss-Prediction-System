
import requests
import pandas as pd
import numpy as np

URL = "http://127.0.0.1:8000/history"

response = requests.get(URL, timeout=15)
response.raise_for_status()

data = response.json()
rows = data.get("readings", [])

if not rows:
    print("No sensor readings found.")
    raise SystemExit

df = pd.DataFrame(rows)

df["received_at"] = pd.to_datetime(
    df["received_at"], utc=True, errors="coerce"
)

df = df.sort_values("received_at").reset_index(drop=True)

print("\nMOTORAI REAL SENSOR DATA REPORT")
print("--------------------------------")

print("API total count:", data.get("count"))
print("Readings received:", len(df))

print("\nTIME RANGE")
print("First:", df["received_at"].iloc[0])
print("Last:", df["received_at"].iloc[-1])

intervals = df["received_at"].diff().dt.total_seconds().dropna()

if len(intervals):
    print("\nSAMPLING INTERVAL")
    print("Average:", round(intervals.mean(), 3), "seconds")
    print("Minimum:", round(intervals.min(), 3), "seconds")
    print("Maximum:", round(intervals.max(), 3), "seconds")

columns = [
    "temperature_c",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "vibration_rms_g",
]

print("\nSENSOR DATA QUALITY")

for col in columns:
    if col not in df.columns:
        print(col, ": MISSING")
        continue

    values = pd.to_numeric(df[col], errors="coerce")

    print(
        col,
        "| Missing:", values.isna().sum(),
        "| Min:", round(values.min(), 5) if values.notna().any() else None,
        "| Max:", round(values.max(), 5) if values.notna().any() else None,
        "| Mean:", round(values.mean(), 5) if values.notna().any() else None,
    )

print("\nDUPLICATE TIMESTAMPS:", df["received_at"].duplicated().sum())

print("\nFIRST 5 CHRONOLOGICAL READINGS")
print(
    df[["received_at"] + columns]
    .head()
    .to_string(index=False)
)

print("\nLAST 5 CHRONOLOGICAL READINGS")
print(
    df[["received_at"] + columns]
    .tail()
    .to_string(index=False)
)

print("\nANALYSIS COMPLETE")
print("No database or dashboard data was modified.")
