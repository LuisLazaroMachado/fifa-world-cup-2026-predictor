from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

# ============================
# Cargar modelo y datos
# ============================

model = joblib.load(DATA / "rf_model.pkl")

teams = pd.read_csv(DATA / "master_dataset.csv")
matches = pd.read_csv(DATA / "group_stage_matches.csv")

teams = teams.set_index("country")

feature_names = model.named_steps["prep"].feature_names_in_

# ============================
# Mapeo de variables
# ============================

mapping = {
    "previous_points": "previous_points",
    "rank_y": "rank_y",
    "rank_max": "rank_max",
    "rating_max": "rating_max",
    "rank_avg": "rank_avg",
    "rating_avg": "rating_avg",
    "rank_min": "rank_min",
    "rating_min": "rating_min",
    "matches_total": "matches_total",
    "matches_home": "matches_home",
    "matches_away": "matches_away",
    "matches_neutral": "matches_neutral",
    "wins": "wins_history",
    "losses": "losses_history",
    "draws": "draws_history",
    "goals_for": "goals_for_history",
    "goals_against": "goals_against_history",
    "is_host": "is_host",
}

# ============================
# Tabla por grupos
# ============================

tables = {}

for group in sorted(matches.group.unique()):

    teams_group = pd.unique(
        pd.concat(
            [
                matches.loc[matches.group == group, "home_team"],
                matches.loc[matches.group == group, "away_team"],
            ]
        )
    )

    tables[group] = {team: 0 for team in teams_group}

# ============================
# Simulación
# ============================

for _, row in matches.iterrows():

    home = row.home_team
    away = row.away_team
    group = row.group

    home_data = teams.loc[home]
    away_data = teams.loc[away]

    sample = {}

    for col in feature_names:

        if col.startswith("home_"):

            key = col.replace("home_", "")
            sample[col] = home_data[mapping[key]]

        elif col.startswith("away_"):

            key = col.replace("away_", "")
            sample[col] = away_data[mapping[key]]

        elif col == "rank_diff":

            sample[col] = away_data["rank_y"] - home_data["rank_y"]

        elif col == "points_diff":

            sample[col] = (
                home_data["total_points"]
                - away_data["total_points"]
            )

        elif col == "rank_change_diff":

            sample[col] = (
                home_data["rank_change"]
                - away_data["rank_change"]
            )

        elif col == "elo_diff":

            sample[col] = (
                home_data["rating"]
                - away_data["rating"]
            )

        elif col == "win_rate_diff":

            sample[col] = (
                home_data["win_rate"]
                - away_data["win_rate"]
            )

        elif col == "goal_diff_diff":

            sample[col] = (
                home_data["goal_difference"]
                - away_data["goal_difference"]
            )

        elif col == "host_advantage":

            sample[col] = (
                int(home_data["is_host"])
                - int(away_data["is_host"])
            )

    X = pd.DataFrame([sample])

    pred = model.predict(X)[0]

    if pred == 1:

        tables[group][home] += 3

    elif pred == 0:

        tables[group][away] += 3

    else:

        tables[group][home] += 1
        tables[group][away] += 1

# ============================
# Mostrar clasificación
# ============================

qualified = []

for group in sorted(tables.keys()):

    print("\n==============================")
    print(f"GRUPO {group}")
    print("==============================")

    ranking = (
        pd.DataFrame(
            tables[group].items(),
            columns=["team", "points"],
        )
        .sort_values(
            ["points", "team"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    print(ranking)

    qualified.extend(ranking.head(2)["team"].tolist())

# ============================
# Clasificados
# ============================

print("\n==============================")
print("CLASIFICADOS")
print("==============================")

for team in qualified:
    print(team)

pd.DataFrame(
    {"team": qualified}
).to_csv(
    DATA / "qualified_teams.csv",
    index=False,
)

print("\nArchivo guardado:")
print(DATA / "qualified_teams.csv")