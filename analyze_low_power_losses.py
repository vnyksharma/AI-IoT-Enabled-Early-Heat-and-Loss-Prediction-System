
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILE = (
    ROOT / "datasets" / "raw"
    / "Three-Phase-Induction-Motor-Design-Data-main"
    / "EMD Data Sheet - Original-200X97 Samples.csv"
)

df = pd.read_csv(FILE)

rating = "Full Load Output(kW)"
loss = "At Full Load : Losses(W)"

low = df[
    (df[rating] >= 0.5) &
    (df[rating] <= 5.0)
].copy()

low = low.sort_values(rating)

print("LOW-POWER MOTOR DATA")
print("Number of rows:", len(low))

print("\nRATING AND LOSS DATA")
print(
    low[[rating, loss]].round(2).to_string(index=False)
)

print("\nSUMMARY")
print(low[[rating, loss]].describe().round(2))

print("\nNEAREST DESIGNS TO OUR 3 HP MOTOR")

target = 3 * 0.746

low["Distance_kW"] = (low[rating] - target).abs()

print(
    low.nsmallest(10, "Distance_kW")[
        [rating, loss, "Distance_kW"]
    ].round(3).to_string(index=False)
)

print("\nANALYSIS COMPLETE")
