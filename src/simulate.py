from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from bracket import KNOCKOUT, ROUND_OF_32, third_place_opponents
from predict import match_proba

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

N_SIMS = 10_000
SEED = 42
HOSTS = {"Mexico", "Canada", "United States"}

# Etapa que alcanza el ganador de cada partido
STAGE_REACHED = (
    {m: "R16" for m in range(73, 89)}
    | {m: "QF" for m in range(89, 97)}
    | {m: "SF" for m in range(97, 101)}
    | {101: "Final", 102: "Final", 104: "Champion"}
)

groups_df = pd.read_csv(DATA_DIR / "world_cup_groups.csv")
GROUP_TEAMS = {g: d["team"].tolist() for g, d in groups_df.groupby("group")}


@lru_cache(maxsize=None)
def group_probs(a: str, b: str) -> tuple:
    """(P gana a, P empate) en fase de grupos. El anfitrión juega como local."""
    if a in HOSTS and b not in HOSTS:
        p = match_proba(a, b, neutral=False)
        return p["home"], p["draw"]
    if b in HOSTS and a not in HOSTS:
        p = match_proba(b, a, neutral=False)
        return p["away"], p["draw"]
    p = match_proba(a, b, neutral=True)
    return p["home"], p["draw"]


@lru_cache(maxsize=None)
def advance_prob(a: str, b: str) -> float:
    """P de que a avance en eliminatoria: si hay empate, se reparte 50/50 (prórroga y penales)."""
    p = match_proba(a, b, neutral=True)
    return p["home"] + p["draw"] / 2


def simulate_groups(rng):
    """Devuelve {grupo: [1º, 2º, 3º, 4º]} y {grupo: puntos del 3º}."""
    standings, third_points = {}, {}
    for g, teams in GROUP_TEAMS.items():
        pts = dict.fromkeys(teams, 0)
        for a, b in combinations(teams, 2):
            pa, pd_ = group_probs(a, b)
            u = rng.random()
            if u < pa:
                pts[a] += 3
            elif u < pa + pd_:
                pts[a] += 1
                pts[b] += 1
            else:
                pts[b] += 3
        # desempate aleatorio (el modelo no simula goles)
        order = sorted(teams, key=lambda t: (pts[t], rng.random()), reverse=True)
        standings[g] = order
        third_points[g] = pts[order[2]]
    return standings, third_points


def simulate_tournament(rng):
    standings, third_points = simulate_groups(rng)
    best_thirds = sorted(third_points, key=lambda g: (third_points[g], rng.random()), reverse=True)[:8]
    thirds = third_place_opponents(best_thirds)

    def resolve(code, match):
        if code == "3":
            return standings[thirds[match]][2]
        return standings[code[1]][int(code[0]) - 1]

    winners, reached = {}, defaultdict(set)
    for m, (a, b) in ROUND_OF_32.items():
        ta, tb = resolve(a, m), resolve(b, m)
        reached["R32"].update((ta, tb))
        winners[m] = ta if rng.random() < advance_prob(ta, tb) else tb
        reached[STAGE_REACHED[m]].add(winners[m])
    for m, (x, y) in KNOCKOUT.items():
        ta, tb = winners[x], winners[y]
        winners[m] = ta if rng.random() < advance_prob(ta, tb) else tb
        reached[STAGE_REACHED[m]].add(winners[m])
    return reached


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    stages = ["R32", "R16", "QF", "SF", "Final", "Champion"]
    counts = {s: defaultdict(int) for s in stages}
    for _ in range(N_SIMS):
        reached = simulate_tournament(rng)
        for s in stages:
            for t in reached[s]:
                counts[s][t] += 1
    teams = groups_df["team"].tolist()
    out = pd.DataFrame({s: [counts[s][t] / N_SIMS for t in teams] for s in stages}, index=teams)
    out.index.name = "team"
    out = out.sort_values("Champion", ascending=False)
    out.reset_index().to_csv(DATA_DIR / "simulation_results.csv", index=False)
    print(f"{N_SIMS} simulaciones")
    print((out.head(12) * 100).round(1).to_string())