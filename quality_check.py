
from pathlib import Path
import pandas as pd

RAW = Path("datasets/raw")
REPORTS = Path("datasets/reports")
REPORTS.mkdir(parents=True, exist_ok=True)

files = [
    RAW / "Three-Phase-Induction-Motor-Design-Data-main"
       / "EMD Data Sheet - Original-200X97 Samples.csv",
    RAW / "Three-Phase-Induction-Motor-Design-Data-main"
       / "EMD DataSheet Variables-interpolated-6000X97 Samples.csv",
]

for file in files:
    print("\n" + "=" * 65)
    print("DATASET:", file.name)

    if not file.exists():
        print("FILE NOT FOUND:", file)
        continue

    df = pd.read_csv(file)

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    print("Shape:", df.shape)
    print("Duplicate rows:", df.duplicated().sum())

    numeric = df.select_dtypes(include="number")

    print("\nConstant columns:")
    print(numeric.columns[numeric.nunique() <= 1].tolist())

    print("\nPotentially suspicious numeric values:")
    for col in numeric.columns:
        if numeric[col].nunique() > 1:
            low = numeric[col].quantile(0.01)
            high = numeric[col].quantile(0.99)
            minimum = numeric[col].min()
            maximum = numeric[col].max()

            if minimum < low - 3 * (high - low) or maximum > high + 3 * (high - low):
                print(
                    f"{col}: min={minimum:.4f}, "
                    f"1%={low:.4f}, 99%={high:.4f}, "
                    f"max={maximum:.4f}"
                )

    print("\nLoss-related columns:")
    for col in df.columns:
        if "loss" in col.lower():
            print(
                f"{col}: min={df[col].min()}, "
                f"max={df[col].max()}, "
                f"unique={df[col].nunique()}"
            )

print("\nQUALITY CHECK COMPLETE")
