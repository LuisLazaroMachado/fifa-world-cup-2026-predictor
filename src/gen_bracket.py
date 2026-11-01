# src/gen_bracket.py
import pandas as pd
from pathlib import Path

DATA = Path("data/processed")

elo = pd.read_csv(DATA / "elo_ratings_wc2026_clean.csv")
elo_latest = elo.sort_values("year").groupby("country").last().reset_index()
elo_dict = dict(zip(elo_latest["country"], elo_latest["rating"]))

def predict_match(h, a):
    eh = elo_dict.get(h, 1500)
    ea = elo_dict.get(a, 1500)
    ph = round(1 / (1 + 10 ** (-(eh - ea) / 400)) * 100, 1)
    pa = round(100 - ph, 1)
    w  = h if ph >= pa else a
    return {"home_team": h, "away_team": a,
            "home_probability": ph, "away_probability": pa, "winner": w}

# Round of 32 (12 partidos → 12 ganadores)
r32 = pd.read_csv(DATA / "round_of_32.csv")[["home_team", "away_team"]]
r32_results = [predict_match(r.home_team, r.away_team) for _, r in r32.iterrows()]
pd.DataFrame(r32_results).to_csv(DATA / "round_of_32.csv", index=False)
winners_r32 = [r["winner"] for r in r32_results]

# Quarterfinals (tomamos top 8 por Elo de los 12 ganadores)
winners_r32_sorted = sorted(
    winners_r32,
    key=lambda t: elo_dict.get(t, 1500),
    reverse=True
)[:8]

qf = [(winners_r32_sorted[i], winners_r32_sorted[i+1]) 
      for i in range(0, 8, 2)]
qf_results = [predict_match(h, a) for h, a in qf]
pd.DataFrame(qf_results).to_csv(DATA / "quarter_matches.csv", index=False)
winners_qf = [r["winner"] for r in qf_results]

# Semifinals
sf = [(winners_qf[0], winners_qf[1]), (winners_qf[2], winners_qf[3])]
sf_results = [predict_match(h, a) for h, a in sf]
pd.DataFrame(sf_results).to_csv(DATA / "semifinal_matches.csv", index=False)
winners_sf = [r["winner"] for r in sf_results]

# Final
final = predict_match(winners_sf[0], winners_sf[1])
pd.DataFrame([final]).to_csv(DATA / "final_match.csv", index=False)
pd.DataFrame([{"team": final["winner"]}]).to_csv(DATA / "champion.csv", index=False)

# Round 16 (4 perdedores del R32 vs 4 equipos extra)
losers_r32 = [r["home_team"] if r["winner"] == r["away_team"] 
              else r["away_team"] for r in r32_results]
losers_sorted = sorted(
    losers_r32,
    key=lambda t: elo_dict.get(t, 1500),
    reverse=True
)[:4]
extra = [t for t in winners_r32 if t not in winners_r32_sorted][:4]
r16_teams = losers_sorted + extra
r16 = [(r16_teams[i], r16_teams[i+1]) for i in range(0, len(r16_teams)-1, 2)]
r16_results = [predict_match(h, a) for h, a in r16]
pd.DataFrame(r16_results).to_csv(DATA / "round16_matches.csv", index=False)

print("✓ Bracket generado sin duplicados.")
print(f"Campeon: {final['winner']}")