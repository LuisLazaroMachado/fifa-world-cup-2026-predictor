from pathlib import Path
import pandas as pd

from elo import compute_elo
from features import add_form_features

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

TRAIN_START = "1990-01-01"
WC2026_START = "2026-06-11"

results = pd.read_csv(DATA_DIR / "results_historical_clean.csv", parse_dates=["date"])

# Elo y forma con TODO el historial (necesitan "calentarse"); luego se recorta.
matches, _ = compute_elo(results)
matches = add_form_features(matches)

matches = matches[
    (matches["date"] >= TRAIN_START) & (matches["date"] < WC2026_START)
].copy()

matches["target"] = 2
matches.loc[matches["home_score"] > matches["away_score"], "target"] = 1
matches.loc[matches["home_score"] < matches["away_score"], "target"] = 0

matches["elo_diff"] = matches["home_elo_pre"] - matches["away_elo_pre"]
matches["neutral"] = matches["neutral"].astype(int)
matches[["form_pts_diff", "form_gd_diff"]] = matches[["form_pts_diff", "form_gd_diff"]].fillna(0)

cols = ["date", "home_team", "away_team", "tournament", "neutral",
        "home_elo_pre", "away_elo_pre", "elo_diff",
        "form_pts_diff", "form_gd_diff", "target"]
matches[cols].to_csv(DATA_DIR / "training_matches.csv", index=False)

print("Filas:", len(matches))
print("Rango de fechas:", matches["date"].min().date(), "->", matches["date"].max().date())
print("\nDistribución del target (1=local, 2=empate, 0=visitante):")
print(matches["target"].value_counts(normalize=True).round(3))
print("\nNulos:", matches[cols].isna().sum().sum())
wc = matches[matches["tournament"] == "FIFA World Cup"]
print("Partidos de Mundial incluidos:", len(wc), "| años:", sorted(wc["date"].dt.year.unique()))