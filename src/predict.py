from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

_bundle = joblib.load(MODEL_DIR / "outcome_model.joblib")
_model = _bundle["model"]
_features = _bundle["features"]
_ratings = pd.read_csv(DATA_DIR / "elo_pre_wc2026.csv").set_index("team")["rating"].to_dict()


def _raw_proba(home: str, away: str, neutral: int) -> np.ndarray:
    """Probabilidades [visitante, local, empate] según el modelo, sin simetrizar."""
    x = pd.DataFrame(
        {"elo_diff_n": [(_ratings[home] - _ratings[away]) / 400], "neutral": [neutral]}
    )[_features]
    return _model.predict_proba(x)[0]


def match_proba(home: str, away: str, neutral: bool = True) -> dict:
    """Probabilidades de victoria local, empate y victoria visitante.

    En cancha neutral el orden local/visitante es arbitrario, así que se
    promedia (A vs B) con (B vs A) para que el resultado no dependa del orden.
    """
    p = _raw_proba(home, away, int(neutral))
    if neutral:
        q = _raw_proba(away, home, 1)
        p = (p + q[[1, 0, 2]]) / 2
    return {"home": float(p[1]), "draw": float(p[2]), "away": float(p[0])}


if __name__ == "__main__":
    for a, b, neutral in [("Spain", "Argentina", True), ("Argentina", "Spain", True),
                          ("France", "Qatar", True), ("Mexico", "South Africa", False)]:
        p = match_proba(a, b, neutral)
        print(f"{a} vs {b} (neutral={neutral}): "
              f"gana {a} {p['home']:.1%} | empate {p['draw']:.1%} | gana {b} {p['away']:.1%}")