
from pathlib import Path
import pandas as pd
import numpy as np

file = Path(
    "datasets/raw/Three-Phase-Induction-Motor-Design-Data-main/"
    "EMD Data Sheet - Original-200X97 Samples.csv"
)

df = pd.read_csv(file)

checks = {
    "Total rotor copper loss": (
        "Total Rotor Copper Loss(W)",
        ["Copper Loss in Bars(W)", "Copper Loss in End Rings(W)"]
    ),
    "No load loss": (
        "No Load Loss(W)",
        ["Core Loss(W)", "Friction and Windage  loss(W)"]
    ),
    "Full load total loss": (
        "At Full Load : Losses(W)",
        ["At Full Load : Input(W)", "At Full Load : Output(W)"]
    )
}

for name, (target, inputs) in checks.items():
    actual = df[target]

    if name == "Full load total loss":
        calculated = df[inputs[0]] - df[inputs[1]]
    else:
        calculated = df[inputs].sum(axis=1)

    difference = actual - calculated

    print("\n" + "=" * 55)
    print(name)
    print("Maximum absolute difference:", difference.abs().max())
    print("Mean absolute difference:", difference.abs().mean())
    print("Matching records:", np.isclose(actual, calculated).sum(), "/", len(df))

print("\nLOSS RELATIONSHIP CHECK COMPLETE")
