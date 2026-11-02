from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="FIFA World Cup 2026 Predictor", page_icon="🏆", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

N_SIMS = 10_000  # debe coincidir con N_SIMS de src/simulate.py

# Resultado real del torneo (fuente: Wikipedia, "2026 FIFA World Cup knockout stage")
REAL_RESULT = {"Spain": "Champion", "Argentina": "Runner-up", "England": "3rd place", "France": "4th place"}

STAGE_NAMES = {
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarterfinals",
    "SF": "Semifinals", "Final": "Final", "Champion": "Champion",
}


@st.cache_data
def load_data():
    sim = pd.read_csv(DATA / "simulation_results.csv")
    evaluation = pd.read_csv(DATA / "evaluation_wc2026.csv")
    groups = pd.read_csv(DATA / "world_cup_groups.csv")
    return sim.merge(groups, on="team"), evaluation


sim, evaluation = load_data()
sim = sim.sort_values("Champion", ascending=False).reset_index(drop=True)
favorite = sim.iloc[0]
real_champion = next(t for t, r in REAL_RESULT.items() if r == "Champion")
top4_are_semifinalists = set(sim.head(4)["team"]) == set(REAL_RESULT)

st.title("🏆 FIFA World Cup 2026 Predictor")
st.caption(
    "Pre-tournament forecast built with Elo ratings, a logistic regression and "
    f"a Monte Carlo simulation of {N_SIMS:,} tournaments."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model favorite", favorite["team"])
c2.metric("Title probability", f"{favorite['Champion'] * 100:.1f}%")
c3.metric("Real champion", real_champion)
c4.metric("Simulations", f"{N_SIMS:,}")

tab_forecast, tab_reality, tab_method = st.tabs(["📊 Forecast", "🔍 Model vs reality", "🧪 Methodology"])

# ----------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Probability of winning the World Cup")
    top = sim.head(15).assign(pct=lambda d: d["Champion"] * 100).iloc[::-1]
    fig = px.bar(top, x="pct", y="team", orientation="h", text=top["pct"].map("{:.1f}%".format))
    fig.update_layout(height=520, xaxis_title="Probability (%)", yaxis_title=None, margin=dict(l=0, r=0, t=10, b=0))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Probability of reaching each round (all 48 teams)")
    table = sim[["team", "group", *STAGE_NAMES]].copy()
    table[list(STAGE_NAMES)] = (table[list(STAGE_NAMES)] * 100).round(1)
    table = table.rename(columns={"team": "Team", "group": "Group", **STAGE_NAMES})
    st.dataframe(table, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------
with tab_reality:
    st.subheader("Forecast vs what actually happened")
    rows = []
    for team, result in REAL_RESULT.items():
        r = sim[sim["team"] == team].iloc[0]
        rows.append({
            "Team": team,
            "Real result": result,
            "Model rank (title odds)": int(sim.index[sim["team"] == team][0]) + 1,
            "P(semifinal)": f"{r['SF'] * 100:.1f}%",
            "P(final)": f"{r['Final'] * 100:.1f}%",
            "P(champion)": f"{r['Champion'] * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if top4_are_semifinalists:
        st.caption("The four most likely champions according to the model were the four real semifinalists.")
    st.caption(
        f"One tournament is a single sample: a {favorite['Champion'] * 100:.0f}% favorite "
        "is expected to win only about 1 in 5 times."
    )

    st.subheader("Match-level accuracy")
    ev = evaluation.rename(columns={"model": "Model", "accuracy": "Accuracy", "log_loss": "Log loss"})
    st.dataframe(ev, hide_index=True, width="stretch")
    st.caption(
        "Computed on the 2026 World Cup matches available in the dataset (until July 4, 2026), "
        "using ratings frozen before the tournament. Lower log loss is better."
    )

# ----------------------------------------------------------------------------
with tab_method:
    st.markdown(
        """
### Pipeline
1. **Data:** ~49,000 international results (1872-2026).
2. **Elo ratings:** computed from scratch for every team (initial 1500, home advantage 100,
   K = 60 World Cup / 40 qualifiers and continental finals / 30 other / 20 friendlies, goal-difference multiplier).
   The rating used for each match is the one *before* it was played.
3. **Model:** logistic regression (rating difference + neutral venue) trained on ~32,000 matches since 1990.
4. **Simulation:** 12 groups, top 2 + 8 best third-placed teams, official round-of-32 bracket and
   third-place assignment table (FIFA regulations, Annex C).

### Validation
- Temporal splits only (train on the past, test on the future).
- 541 competitive matches of 2022: log loss 0.932 vs 1.049 for historical frequencies.
- World Cups 2018 and 2022 (64 matches each): differences between models are within noise.
- Richer models (recent form, gradient boosting) did **not** improve results, so the simple model was kept.

### Assumptions and limitations
- Group and third-place tie-breakers are random (the model predicts outcomes, not goals).
- Knockout draws are resolved 50/50 (extra time and penalties).
- Host advantage applies only in the group stage.
- Ratings are not updated during the tournament; injuries and squads are not modelled.
        """
    )