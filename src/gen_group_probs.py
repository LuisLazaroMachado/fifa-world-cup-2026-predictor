from pathlib import Path
import pandas as pd
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

# Cargar modelo y ELO (datos reales)
model = joblib.load(DATA / "rf_model.pkl")
elo   = pd.read_csv(DATA / "elo_ratings_wc2026_clean.csv")
groups = pd.read_csv(DATA / "world_cup_groups.csv")
matches = pd.read_csv(DATA / "group_stage_matches.csv")

# Tomar ELO más reciente por equipo
elo_latest = elo.sort_values("year").groupby("country").last().reset_index()
elo_dict   = dict(zip(elo_latest["country"], elo_latest["rating"]))

# Calcular puntos por grupo simulando cada partido
team_points = {t: 0 for t in groups["team"]}
team_prob   = {}

for _, match in matches.iterrows():
    home = match["home_team"]
    away = match["away_team"]
    
    elo_h = elo_dict.get(home, 1500)
    elo_a = elo_dict.get(away, 1500)
    
    # Probabilidad basada en ELO (fórmula estándar)
    diff = elo_h - elo_a
    prob_home = 1 / (1 + 10 ** (-diff / 400))
    prob_away = 1 - prob_home
    
    # Puntos esperados
    team_points[home] = team_points.get(home, 0) + prob_home * 3
    team_points[away] = team_points.get(away, 0) + prob_away * 3

# Calcular probabilidad de clasificación por grupo
for g in groups["group"].unique():
    teams = groups[groups["group"] == g]["team"].tolist()
    pts   = {t: team_points.get(t, 0) for t in teams}
    total = sum(pts.values()) or 1
    
    # Top 2 clasifican directamente
    sorted_teams = sorted(pts.items(), key=lambda x: x[1], reverse=True)
    for i, (team, pt) in enumerate(sorted_teams):
        if i == 0:
            prob = 85 + (pt/total) * 10
        elif i == 1:
            prob = 55 + (pt/total) * 10
        elif i == 2:
            prob = 25 + (pt/total) * 10
        else:
            prob = 5  + (pt/total) * 10
        team_prob[team] = min(prob, 99)

# Guardar
out = pd.DataFrame([
    {"team": t, "probability": p} 
    for t, p in team_prob.items()
])
out.to_csv(DATA / "group_probability.csv", index=False)
print("✓ Probabilidades reales generadas:")
print(out.sort_values("probability", ascending=False).head(10))