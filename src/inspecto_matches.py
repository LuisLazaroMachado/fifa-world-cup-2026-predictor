from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

matches = pd.read_csv(DATA_DIR / "results_historical_clean.csv")

# Solo Mundial 2026
matches = matches[
    (matches["tournament"] == "FIFA World Cup") &
    (pd.to_datetime(matches["date"]).dt.year == 2026)
]

print(matches.shape)
print()

print(matches.columns.tolist())
print()

print(matches.head(20))

print()
print("Número de partidos:", len(matches))

print("\nPartidos por fecha:\n")
print(matches["date"].value_counts().sort_index())

print("\nÚltimos 20 partidos:\n")
print(matches.tail(20)[["date","home_team","away_team"]])