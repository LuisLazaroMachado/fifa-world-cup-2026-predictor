import numpy as np
import pandas as pd


def add_form_features(matches: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Forma reciente de cada equipo (últimos n partidos ANTES del partido actual)."""
    m = matches.reset_index(drop=True).copy()
    home = pd.DataFrame({"match_id": m.index, "date": m["date"], "team": m["home_team"],
                         "gf": m["home_score"], "ga": m["away_score"], "side": "home"})
    away = pd.DataFrame({"match_id": m.index, "date": m["date"], "team": m["away_team"],
                         "gf": m["away_score"], "ga": m["home_score"], "side": "away"})
    long = pd.concat([home, away]).sort_values(["team", "date", "match_id"])
    long["pts"] = np.where(long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0))
    long["gd"] = long["gf"] - long["ga"]

    grouped = long.groupby("team")
    for col in ("pts", "gd"):
        long[f"form_{col}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(n, min_periods=3).mean()   # shift(1): excluye el partido actual
        )

    for side in ("home", "away"):
        part = long[long["side"] == side].set_index("match_id")[["form_pts", "form_gd"]]
        m[f"{side}_form_pts"] = part["form_pts"]
        m[f"{side}_form_gd"] = part["form_gd"]

    m["form_pts_diff"] = m["home_form_pts"] - m["away_form_pts"]
    m["form_gd_diff"] = m["home_form_gd"] - m["away_form_gd"]
    return m