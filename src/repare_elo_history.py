import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

elo = pd.read_csv(
    ROOT_DIR / "data" / "raw" / "world_cup_historical_stats_48_teams" / "elo_ratings_wc2026.csv"
)

print(elo.shape)
print(elo.columns.tolist())
print("Años:", elo["year"].min(), elo["year"].max())
print("Selecciones:", elo["country"].nunique())