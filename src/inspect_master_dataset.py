from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

df = pd.read_csv(DATA_DIR / "master_dataset.csv")

print(df.shape)
print(df.columns.tolist())

print("\nPrimeras filas:")
print(df.head())

print("\nCountry:")
print(df["country"].tolist())