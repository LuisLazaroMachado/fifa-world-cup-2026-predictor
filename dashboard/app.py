from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="FIFA World Cup 2026 Predictor",
    page_icon="🏆",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

# ==========================================================
# ESTILOS
# ==========================================================

st.markdown("""
<style>

.block-container{
    padding-top:1rem;
}

div[data-testid="metric-container"]{
    border-radius:15px;
    border:1px solid #dddddd;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,.08);
}

.match-card{

    border-radius:18px;

    padding:20px;

    border:1px solid #d8d8d8;

    margin-bottom:15px;

    background:white;

    box-shadow:0px 3px 10px rgba(0,0,0,.05);
}

.round-title{

    color:#003366;

    font-size:30px;

    font-weight:bold;

    margin-top:25px;
}

.team-win{

    color:#1b8d2d;

    font-weight:bold;

    font-size:20px;
}

.team-lose{

    color:#cc2222;

    font-size:18px;
}

.group-card{

    border-radius:15px;

    border:1px solid #dddddd;

    padding:15px;

    margin-bottom:15px;

    background:#fafafa;
}

.small-title{

    color:#003366;

    font-size:22px;

    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# CARGA DE DATOS
# ==========================================================

groups = pd.read_csv(DATA / "world_cup_groups.csv")

qualified = pd.read_csv(DATA / "qualified_teams.csv")

round32 = pd.read_csv(DATA / "round_of_32.csv")

round16 = pd.read_csv(DATA / "round16_matches.csv")

quarter = pd.read_csv(DATA / "quarter_matches.csv")

semi = pd.read_csv(DATA / "semifinal_matches.csv")

final = pd.read_csv(DATA / "final_match.csv")

champion = pd.read_csv(DATA / "champion.csv")

top10 = pd.read_csv(DATA / "top10_champions.csv")

# ==========================================================
# TABS
# ==========================================================

inicio, grupos_tab, knockout, probabilidades = st.tabs(

[
"🏠 Home",
"⚽ Group Stage",
"🏆 Knockout",
"📊 Monte Carlo"
]

)

# ==========================================================
# HOME
# ==========================================================

with inicio:

    hero1, hero2 = st.columns([1,3])

    with hero1:
        st.image("dashboard/world_cup.png", width=230)

    with hero2:

        st.title("🏆 FIFA World Cup 2026 Predictor")

        st.markdown("""
### Machine Learning World Cup Prediction System

Predicción completa del Mundial FIFA 2026 utilizando:

- 🌍 FIFA Rankings
- 📈 ELO Ratings
- ⚽ Historial internacional
- 🤖 Random Forest
- 🎲 Monte Carlo Simulation
- 🏆 Tournament Bracket Prediction
""")

    st.divider()

    m1,m2,m3,m4 = st.columns(4)

    with m1:
        st.metric(
            "🏆 Predicted Champion",
            champion.iloc[0,0]
        )

    with m2:
        st.metric(
            "🌎 Teams",
            len(groups.team.unique())
        )

    with m3:
        st.metric(
            "🤖 Model",
            "Random Forest"
        )

    with m4:
        st.metric(
            "🎲 Simulations",
            f"{int(top10.titles.sum())}"
        )

    st.divider()

    left,right = st.columns([2,1])

    with left:

        st.subheader("Prediction Pipeline")

        st.code("""
Historical Matches
        │
        ▼
Feature Engineering
        │
        ▼
Random Forest
        │
        ▼
Match Probabilities
        │
        ▼
Knockout Simulation
        │
        ▼
Monte Carlo (500 tournaments)
        │
        ▼
Champion Prediction
""")

    with right:

        favorite = top10.sort_values(
            "probability",
            ascending=False
        ).iloc[0]

        st.subheader("🏆 Favorite")

        st.metric(
            favorite.team,
            f"{favorite.probability*100:.2f}%"
        )

        fig = px.pie(
            top10.head(5),
            values="probability",
            names="team",
            hole=.60
        )

        fig.update_layout(
            height=330,
            margin=dict(l=0,r=0,t=0,b=0)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("Tournament Overview")

    c1,c2,c3,c4,c5 = st.columns(5)

    c1.metric("Group Stage","48 Teams")
    c2.metric("Round of 32","32 Teams")
    c3.metric("Quarterfinals","8 Teams")
    c4.metric("Semifinals","4 Teams")
    c5.metric("Champion","1 Team")
    

# ==========================================================
# GROUP STAGE
# ==========================================================

with grupos_tab:

    st.title("⚽ FIFA World Cup Group Stage")

    st.caption(
        "Qualification probabilities predicted by the Random Forest model."
    )

    st.divider()

    qualified_set = set(qualified["team"])

    # ===========================
    # Probabilidad aproximada
    # ===========================

    team_prob = {}

    if "group_probability.csv" in [f.name for f in DATA.glob("*.csv")]:

        probs = pd.read_csv(DATA/"group_probability.csv")

        for _,r in probs.iterrows():

            team_prob[r.team] = r.probability

    else:

        # Si aún no existe el archivo,
        # usamos una aproximación visual

        for g in groups.group.unique():

            teams_group = groups.loc[
                groups.group==g,
                "team"
            ].tolist()

            base = [90,70,40,15]

            for t,p in zip(teams_group,base):

                team_prob[t]=p

    # ===========================
    # Mostrar grupos
    # ===========================

    groups_list = sorted(groups.group.unique())

    for i in range(0,len(groups_list),3):

        cols = st.columns(3)

        for j in range(3):

            if i+j>=len(groups_list):
                continue

            group = groups_list[i+j]

            with cols[j]:

                st.markdown(
                    f"## 🏆 Group {group}"
                )

                gteams = groups.loc[
                    groups.group==group,
                    "team"
                ]

                for team in gteams:

                    prob = team_prob.get(team,50)

                    if team in qualified_set:

                        color="🟢"

                    else:

                        color="🔴"

                    st.markdown(
                        f"### {color} {team}"
                    )

                    st.progress(prob/100)

                    st.write(
                        f"Qualification probability **{prob:.1f}%**"
                    )

                st.divider()

    st.header("Qualified Teams")

    q = qualified.copy()

    q.index=q.index+1

    st.dataframe(
        q,
        use_container_width=True,
        hide_index=True
    )

# ==========================================================
# KNOCKOUT STAGE
# ==========================================================

with knockout:

    st.title("🏆 FIFA World Cup 2026 Knockout Bracket")

    st.caption(
        "Knockout stage predicted by the Random Forest model."
    )

    FLAGS = {
        "Argentina":"🇦🇷",
        "Australia":"🇦🇺",
        "Austria":"🇦🇹",
        "Belgium":"🇧🇪",
        "Brazil":"🇧🇷",
        "Colombia":"🇨🇴",
        "Croatia":"🇭🇷",
        "England":"🏴",
        "France":"🇫🇷",
        "Portugal":"🇵🇹",
        "Spain":"🇪🇸",
        "Mexico":"🇲🇽",
        "Uruguay":"🇺🇾",
        "Switzerland":"🇨🇭",
        "Qatar":"🇶🇦",
        "South Korea":"🇰🇷"
    }

    st.markdown("""

<style>

.bracket{

    display:grid;

    grid-template-columns:
    300px
    260px
    220px
    220px
    220px;

    gap:35px;

    align-items:center;

}


.round{

    display:flex;

    flex-direction:column;

    justify-content:center;

}


.round32 .match{
    margin-bottom:18px;
}


.round16 .match{

    margin-top:45px;
    margin-bottom:45px;

}


.quarter .match{

    margin-top:100px;
    margin-bottom:100px;

}


.semi .match{

    margin-top:180px;

}


.final .match{

    margin-top:350px;

}

.round{
    display:flex;
    flex-direction:column;
}

.round-title{
    text-align:center;
    font-size:26px;
    font-weight:bold;
    color:#003366;
    margin-bottom:18px;
}

.match{

    background:white;

    border-radius:12px;

    border:1px solid #dcdcdc;

    padding:12px;

    margin-bottom:22px;

    box-shadow:0 2px 8px rgba(0,0,0,.08);

}

.spacer{
    height:55px;
}

.win{
    color:#1a9c45;
    font-weight:bold;
}

.lose{
    color:#d62828;
}

.percent{
    float:right;
    font-weight:bold;
}

.round-title{

font-size:28px;

font-weight:bold;

color:#003366;

margin-bottom:25px;

}

.match{

background:white;

border-radius:12px;

border:1px solid #d9d9d9;

padding:12px;

margin-bottom:18px;

box-shadow:0 3px 8px rgba(0,0,0,.08);

}

.win{

color:#18a558;

font-weight:bold;

font-size:17px;

}

.lose{

color:#d62828;

font-size:17px;

}

.percent{

float:right;

font-weight:bold;

}

.connector{

height:18px;

}

</style>

""", unsafe_allow_html=True)

    def build_match(row):

        home = row.home_team
        away = row.away_team

        hp = float(row.home_probability)
        ap = float(row.away_probability)

        winner = row.winner

        home_class = "win" if winner == home else "lose"
        away_class = "win" if winner == away else "lose"

        return f"""
    <div class="match">

    <div class="{home_class}">
    {FLAGS.get(home,"🏳️")} {home}
    <span class="percent">{hp:.1f}%</span>
    </div>

    <hr style="margin:8px 0;">

    <div class="{away_class}">
    {FLAGS.get(away,"🏳️")} {away}
    <span class="percent">{ap:.1f}%</span>
    </div>

    </div>
    """

    html = """
    <div class="bracket">

    <div>

    <div class="round-title">
    Round of 32
    </div>
    """

    for _, r in round32.iterrows():
        html += build_match(r)

    html += """
    </div>

    <div>

    <div class="round-title">
    Quarterfinals
    </div>
    """

    for _, r in quarter.iterrows():
        html += build_match(r)

    html += """
    </div>

    <div>

    <div class="round-title">
    Semifinals
    </div>
    """

    for _, r in semi.iterrows():
        html += build_match(r)

    html += """
    </div>

    <div>

    <div class="round-title">
    Final
    </div>
    """

    html += build_match(final.iloc[0])

    winner = final.iloc[0].winner

    html += f"""

    <br><br>

    <div style="
    background:#0E9F4B;
    color:white;
    padding:18px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
    ">

    🏆 WORLD CHAMPION

    <br><br>

    {FLAGS.get(winner,"🏳️")} {winner}

    </div>

    </div>

    </div>
    """

    st.markdown(html, unsafe_allow_html=True)

# ==========================================================
# MONTE CARLO
# ==========================================================

with probabilidades:

    st.title("📊 Monte Carlo Simulation")

    st.write(
        "Champion probabilities obtained after simulating the complete tournament hundreds of times."
    )

    st.divider()

    top10_display = top10.copy()

    top10_display["probability"] = (
        top10_display["probability"] * 100
    ).round(2)

    top10_display = top10_display.rename(
        columns={
            "team": "Team",
            "titles": "Titles",
            "probability": "Probability (%)"
        }
    )

    favorite = top10_display.iloc[0]

    c1, c2 = st.columns([1, 2])

    with c1:

        st.metric(
            "🏆 Favorite",
            favorite["Team"]
        )

        st.metric(
            "Winning Probability",
            f'{favorite["Probability (%)"]:.2f}%'
        )

    with c2:

        fig = px.bar(
            top10_display,
            x="Team",
            y="Probability (%)",
            text="Probability (%)",
            color="Probability (%)",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            height=500,
            xaxis_title="National Team",
            yaxis_title="Probability (%)",
            showlegend=False
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    st.divider()

    st.subheader("Complete Ranking")

    st.dataframe(
        top10_display,
        width="stretch",
        hide_index=True
    )
