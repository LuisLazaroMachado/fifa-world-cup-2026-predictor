from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

INITIAL_RATING = 1500.0
TEAM_ALIASES = {"Czech Republic": "Czechia"}
HOME_ADVANTAGE = 100.0

CONTINENTAL_FINALS = {
    "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "Gold Cup", "Confederations Cup",
}


def k_factor(tournament: str) -> int:
    """Importancia del partido: más peso a torneos grandes."""
    if tournament == "FIFA World Cup":
        return 60
    if tournament in CONTINENTAL_FINALS or "qualification" in tournament:
        return 40
    if tournament == "Friendly":
        return 20
    return 30


def goal_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def compute_elo(results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Devuelve, por partido, el Elo de cada equipo ANTES de jugarse."""
    results = results.replace({"home_team": TEAM_ALIASES, "away_team": TEAM_ALIASES})
    results = results.sort_values("date").reset_index(drop=True)
    ratings = {}
    home_pre, away_pre = [], []

    for r in results.itertuples():
        rh = ratings.get(r.home_team, INITIAL_RATING)
        ra = ratings.get(r.away_team, INITIAL_RATING)
        home_pre.append(rh)   # se guarda ANTES de actualizar: sin fuga
        away_pre.append(ra)

        adv = 0.0 if r.neutral else HOME_ADVANTAGE
        expected_home = 1 / (1 + 10 ** (-((rh + adv) - ra) / 400))

        if r.home_score > r.away_score:
            actual = 1.0
        elif r.home_score == r.away_score:
            actual = 0.5
        else:
            actual = 0.0

        delta = (
            k_factor(r.tournament)
            * goal_multiplier(abs(r.home_score - r.away_score))
            * (actual - expected_home)
        )
        ratings[r.home_team] = rh + delta
        ratings[r.away_team] = ra - delta

    out = results.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    return out, ratings


if __name__ == "__main__":
    results = pd.read_csv(DATA_DIR / "results_historical_clean.csv", parse_dates=["date"])
    print("dtype neutral:", results["neutral"].dtype)
    matches, ratings = compute_elo(results)
    print("\nTop 10 Elo tras el último partido:")
    print(pd.Series(ratings).sort_values(ascending=False).head(10).round(0))
    print("\nEquipos con rating:", len(ratings))
    print("Nulos en elo_pre:", matches[["home_elo_pre", "away_elo_pre"]].isna().sum().sum())