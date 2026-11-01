from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

model = joblib.load(DATA / "rf_model.pkl")

teams = pd.read_csv(DATA / "master_dataset.csv")
teams = teams.set_index("country")

feature_names = model.named_steps["prep"].feature_names_in_

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


def predict_match(home, away):

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

    proba = model.predict_proba(X)[0]

    home_prob = proba[1]
    away_prob = proba[0]

    total = home_prob + away_prob

    home_prob /= total
    away_prob /= total

    winner = np.random.choice(
        [home, away],
        p=[home_prob, away_prob]
    )

    return winner, home_prob, away_prob


def simulate_round(file_name, output_name):

    matches = pd.read_csv(DATA / file_name)

    winners = []
    results = []

    for _, row in matches.iterrows():

        winner, home_prob, away_prob = predict_match(
            row.home_team,
            row.away_team
        )

        results.append({
            "home_team": row.home_team,
            "away_team": row.away_team,
            "home_probability": round(home_prob * 100, 2),
            "away_probability": round(away_prob * 100, 2),
            "winner": winner
        })

        winners.append(winner)

    pd.DataFrame(
        {"team": winners}
    ).to_csv(
        DATA / output_name,
        index=False
    )

    if output_name == "round16_winners.csv":
        match_file = "round16_matches.csv"

    elif output_name == "quarter_winners.csv":
        match_file = "quarter_matches.csv"

    elif output_name == "finalists.csv":
        match_file = "semifinal_matches.csv"

    else:
        match_file = output_name.replace("_winners.csv", "_matches.csv")

    pd.DataFrame(results).to_csv(
        DATA / match_file,
        index=False
    )

    return winners

def simulate_world_cup(verbose=True):
    if verbose:
        print("\nROUND OF 32\n")

    r16 = simulate_round(
        "round_of_32.csv",
        "round16_winners.csv"
    )

    quarter = []

    for i in range(0, len(r16), 2):
        quarter.append([r16[i], r16[i + 1]])

    quarter = pd.DataFrame(
        quarter,
        columns=["home_team", "away_team"]
    )

    quarter.to_csv(
        DATA / "quarterfinals.csv",
        index=False
    )
    if verbose:
        print("\nQUARTERFINALS\n")

    sf = simulate_round(
        "quarterfinals.csv",
        "quarter_winners.csv"
    )

    semi = []

    for i in range(0, len(sf), 2):
        semi.append([sf[i], sf[i + 1]])

    semi = pd.DataFrame(
        semi,
        columns=["home_team", "away_team"]
    )

    semi.to_csv(
        DATA / "semifinals.csv",
        index=False
    )
    if verbose:
        print("\nSEMIFINALS\n")

    finalists = simulate_round(
        "semifinals.csv",
        "finalists.csv"
    )

    final = pd.DataFrame(
        [[finalists[0], finalists[1]]],
        columns=["home_team", "away_team"]
    )

    final.to_csv(
        DATA / "final.csv",
        index=False
    )
    if verbose:
        print("\nFINAL\n")

    winner, home_prob, away_prob = predict_match(
        finalists[0],
        finalists[1]
    )

    champion = winner

    pd.DataFrame(
        {
            "home_team": [finalists[0]],
            "away_team": [finalists[1]],
            "home_probability": [round(home_prob * 100, 2)],
            "away_probability": [round(away_prob * 100, 2)],
            "winner": [winner]
        }
    ).to_csv(
        DATA / "final_match.csv",
        index=False
    )
    if verbose:
        print("\n======================")
        print("WORLD CHAMPION")
        print("======================")
        print(champion)

    pd.DataFrame(
        {"champion": [champion]}
    ).to_csv(
        DATA / "champion.csv",
        index=False
    )
    return champion

if __name__ == "__main__":
    simulate_world_cup()