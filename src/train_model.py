from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from elo import compute_elo

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

WC2026_START = "2026-06-11"
FEATURES = ["elo_diff_n", "neutral"]

# 1) Modelo: regresión logística entrenada con todo el pasado previo al torneo
df = pd.read_csv(DATA_DIR / "training_matches.csv", parse_dates=["date"])
df = df[df["date"] < WC2026_START].copy()
df["elo_diff_n"] = df["elo_diff"] / 400

model = LogisticRegression(max_iter=1000).fit(df[FEATURES], df["target"])
joblib.dump({"model": model, "features": FEATURES}, MODEL_DIR / "outcome_model.joblib")

print("Partidos de entrenamiento:", len(df))
print("Clases:", model.classes_.tolist(), "(0=visitante, 1=local, 2=empate)")
print("Coeficientes (elo_diff_n, neutral) por clase:")
print(pd.DataFrame(model.coef_, index=model.classes_, columns=FEATURES).round(3))

# 2) Ratings Elo al inicio del torneo (sin partidos del Mundial 2026)
results = pd.read_csv(DATA_DIR / "results_historical_clean.csv", parse_dates=["date"])
_, ratings = compute_elo(results[results["date"] < WC2026_START])
elo_df = (pd.Series(ratings, name="rating").rename_axis("team").reset_index()
          .sort_values("rating", ascending=False))
elo_df.to_csv(DATA_DIR / "elo_pre_wc2026.csv", index=False)

print("\nTop 10 Elo al 10-jun-2026:")
print(elo_df.head(10).round(0).to_string(index=False))

# 3) ¿Están los 48 equipos del Mundial con el mismo nombre en el Elo?
groups = pd.read_csv(DATA_DIR / "world_cup_groups.csv")
missing = sorted(set(groups["team"]) - set(elo_df["team"]))
print("\nEquipos del Mundial sin rating (problema de nombres):", missing)