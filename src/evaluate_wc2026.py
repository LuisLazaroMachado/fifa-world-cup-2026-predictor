from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from predict import match_proba

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
WC2026_START = "2026-06-11"

aliases = {"Czech Republic": "Czechia"}
results = pd.read_csv(DATA_DIR / "results_historical_clean.csv", parse_dates=["date"])
results = results.replace({"home_team": aliases, "away_team": aliases})
wc = results[
    (results["tournament"] == "FIFA World Cup") & (results["date"] >= WC2026_START)
].copy()

# 1 = gana local, 0 = gana visitante, 2 = empate (igual que en el entrenamiento)
wc["target"] = np.select(
    [wc["home_score"] > wc["away_score"], wc["home_score"] < wc["away_score"]], [1, 0], default=2
)

# Probabilidades con el modelo y los ratings de ANTES del torneo (sin actualizar durante el torneo)
proba = np.array([
    [p["away"], p["home"], p["draw"]]
    for p in (match_proba(r.home_team, r.away_team, neutral=bool(r.neutral)) for r in wc.itertuples())
])

ratings = pd.read_csv(DATA_DIR / "elo_pre_wc2026.csv").set_index("team")["rating"]
higher_elo = (wc["home_team"].map(ratings) > wc["away_team"].map(ratings)).astype(int)

train = pd.read_csv(DATA_DIR / "training_matches.csv")
prior = train["target"].value_counts(normalize=True).sort_index().to_numpy()  # orden 0,1,2

y = wc["target"].to_numpy()
print(f"Partidos del Mundial 2026 en los datos: {len(wc)} "
      f"({wc['date'].min().date()} -> {wc['date'].max().date()})")
print(f"Distribución real (0=visitante, 1=local, 2=empate): {np.bincount(y, minlength=3).tolist()}\n")

summary = pd.DataFrame(
    {
        "accuracy": [accuracy_score(y, proba.argmax(axis=1)), accuracy_score(y, higher_elo),
                     accuracy_score(y, np.ones(len(y), dtype=int)), np.nan],
        "log_loss": [log_loss(y, proba, labels=[0, 1, 2]), np.nan, np.nan,
                     log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2])],
    },
    index=["Modelo (logística + Elo)", "Gana el de mayor Elo", "Siempre gana el local",
           "Solo frecuencias históricas"],
)
summary.round(3).rename_axis("model").reset_index().to_csv(DATA_DIR / "evaluation_wc2026.csv", index=False)
print(summary.round(3).to_string())