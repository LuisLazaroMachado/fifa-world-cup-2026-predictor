from pathlib import Path
import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# ==========================
# Cargar modelo y datos
# ==========================

model = joblib.load(DATA_DIR / "rf_model.pkl")
master = pd.read_csv(DATA_DIR / "master_dataset.csv")


def get_team(team):

    row = master.loc[
        master["country"] == team
    ]

    if len(row) == 0:
        raise ValueError(f"No existe {team}")

    return row.iloc[0]


def build_features(home, away):

    h = get_team(home)
    a = get_team(away)

    x = {}

    x["home_previous_points"] = h["previous_points"]
    x["away_previous_points"] = a["previous_points"]

    x["home_rank_y"] = h["rank_y"]
    x["home_rank_max"] = h["rank_max"]
    x["home_rating_max"] = h["rating_max"]
    x["home_rank_avg"] = h["rank_avg"]
    x["home_rating_avg"] = h["rating_avg"]
    x["home_rank_min"] = h["rank_min"]
    x["home_rating_min"] = h["rating_min"]

    x["home_matches_total"] = h["matches_total"]
    x["home_matches_home"] = h["matches_home"]
    x["home_matches_away"] = h["matches_away"]
    x["home_matches_neutral"] = h["matches_neutral"]

    x["home_wins"] = h["wins_elo"]
    x["home_losses"] = h["losses_elo"]
    x["home_draws"] = h["draws_elo"]

    x["home_goals_for"] = h["goals_for_elo"]
    x["home_goals_against"] = h["goals_against_elo"]

    x["home_is_host"] = h["is_host"]

    x["away_rank_y"] = a["rank_y"]
    x["away_rank_max"] = a["rank_max"]
    x["away_rating_max"] = a["rating_max"]
    x["away_rank_avg"] = a["rank_avg"]
    x["away_rating_avg"] = a["rating_avg"]
    x["away_rank_min"] = a["rank_min"]
    x["away_rating_min"] = a["rating_min"]

    x["away_matches_total"] = a["matches_total"]
    x["away_matches_home"] = a["matches_home"]
    x["away_matches_away"] = a["matches_away"]
    x["away_matches_neutral"] = a["matches_neutral"]

    x["away_wins"] = a["wins_elo"]
    x["away_losses"] = a["losses_elo"]
    x["away_draws"] = a["draws_elo"]

    x["away_goals_for"] = a["goals_for_elo"]
    x["away_goals_against"] = a["goals_against_elo"]

    x["away_is_host"] = a["is_host"]

    x["rank_diff"] = a["rank_y"] - h["rank_y"]

    x["points_diff"] = (
        h["total_points"] -
        a["total_points"]
    )

    x["rank_change_diff"] = (
        h["rank_change"] -
        a["rank_change"]
    )

    x["elo_diff"] = (
        h["rating"] -
        a["rating"]
    )

    x["win_rate_diff"] = (
        h["win_rate"] -
        a["win_rate"]
    )

    x["goal_diff_diff"] = (
        h["goal_difference"] -
        a["goal_difference"]
    )

    x["host_advantage"] = (
        h["is_host"] -
        a["is_host"]
    )

    return pd.DataFrame([x])


def predict_match(home, away):

    X = build_features(home, away)

    pred = model.predict(X)[0]

    prob = model.predict_proba(X)[0]

    clases = model.classes_

    probabilidades = dict(zip(clases, prob))

    if pred == 1:
        winner = home
    elif pred == 0:
        winner = away
    else:
        winner = "Draw"

    print()
    print(home, "vs", away)
    print("----------------------")
    print("Ganador:", winner)
    print()
    print("Probabilidades:")
    print(probabilidades)

    return winner


if __name__ == "__main__":

    predict_match(
        "Argentina",
        "Spain"
    )