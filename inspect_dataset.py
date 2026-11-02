
import pandas as pd
import os

# Dataset location
DATASET_FOLDER = os.path.join(
    "datasets",
    "raw",
    "Three-Phase-Induction-Motor-Design-Data-main"
)

ORIGINAL_FILE = os.path.join(
    DATASET_FOLDER,
    "EMD Data Sheet - Original-200X97 Samples.csv"
)

EXPANDED_FILE = os.path.join(
    DATASET_FOLDER,
    "EMD DataSheet Variables-interpolated-6000X97 Samples.csv"
)

def inspect_dataset(file_path, dataset_name):

    print("\n" + "=" * 70)
    print(f"INSPECTING: {dataset_name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print("\n1. DATASET SHAPE")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    print("\n2. COLUMN NAMES")
    for index, column in enumerate(df.columns, 1):
        print(f"{index}. {column}")

    print("\n3. DATA TYPES")
    print(df.dtypes)

    print("\n4. MISSING VALUES")
    print(df.isnull().sum().to_string())

    print("\n5. DUPLICATE ROWS")
    print(df.duplicated().sum())

    print("\n6. STATISTICAL SUMMARY")
    print(df.describe(include="all").to_string())

    print("\n7. FIRST FIVE RECORDS")
    print(df.head().to_string())

    # Save inspection report
    report_name = dataset_name.lower().replace(" ", "_")
    report_path = os.path.join(
        "datasets",
        "reports",
        report_name + "_report.txt"
    )

    os.makedirs("datasets/reports", exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"DATASET: {dataset_name}\n")
        file.write(f"SHAPE: {df.shape}\n\n")
        file.write("COLUMNS:\n")
        file.write("\n".join(df.columns.astype(str)))
        file.write("\n\nDATA TYPES:\n")
        file.write(df.dtypes.to_string())
        file.write("\n\nMISSING VALUES:\n")
        file.write(df.isnull().sum().to_string())
        file.write("\n\nSTATISTICAL SUMMARY:\n")
        file.write(df.describe(include="all").to_string())

    print("\nReport saved:", report_path)

    return df


if __name__ == "__main__":

    original = inspect_dataset(
        ORIGINAL_FILE,
        "Original Motor Dataset"
    )

    expanded = inspect_dataset(
        EXPANDED_FILE,
        "Expanded Motor Dataset"
    )

    print("\nDATASET INSPECTION COMPLETED!")
