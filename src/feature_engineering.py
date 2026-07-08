# ==========================================================
# FIFA World Cup 2026 Predictor
# Feature Engineering
#
# Este script integra los datasets procesados y genera el
# dataset maestro para los modelos MLP y LSTM.
# ==========================================================

from pathlib import Path
import pandas as pd

# ==========================================================
# Definir rutas del proyecto
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# ==========================================================
# Cargar datasets procesados
# ==========================================================

ranking = pd.read_csv(DATA_DIR / "fifa_world_ranking_clean.csv")
players = pd.read_csv(DATA_DIR / "players_clean.csv")
results = pd.read_csv(DATA_DIR / "results_historical_clean.csv")
elo = pd.read_csv(DATA_DIR / "elo_ratings_wc2026_clean.csv")

print("Ranking :", ranking.shape)
print("Players :", players.shape)
print("Results :", results.shape)
print("ELO      :", elo.shape)

# ==========================================================
# Preparar variables de jugadores
# ==========================================================

# Convertir la edad de formato "26-138" a años decimales
age_split = players["age"].str.split("-", expand=True)

players["age"] = (
    age_split[0].astype(int) +
    age_split[1].astype(int) / 365
)

# ==========================================================
# Generar estadísticas agregadas por selección
# ==========================================================

players_summary = (
    players.groupby("team_country", as_index=False)
    .agg(
        num_players=("player", "count"),
        avg_age=("age", "mean"),
        avg_minutes=("minutes", "mean"),
        avg_goals=("goals", "mean"),
        avg_assists=("assists", "mean"),
        avg_shots=("shots", "mean"),
        avg_points_per_game=("points_per_game", "mean")
    )
)

# ==========================================================
# Cantidad de jugadores por posición
# ==========================================================

positions = pd.crosstab(
    players["team_country"],
    players["position"]
).reset_index()

players_summary = players_summary.merge(
    positions,
    on="team_country",
    how="left"
)

print("Players Summary")
print(players_summary.shape)

print("\nPlayers Summary:")
print(players_summary.head())

print("\nColumnas:")
print(players_summary.columns.tolist())

# ==========================================================
# Generar estadísticas históricas por selección
# ==========================================================

# Partidos como local
home = pd.DataFrame({
    "country": results["home_team"],
    "goals_for": results["home_score"],
    "goals_against": results["away_score"],
    "result": (
        results["home_score"] > results["away_score"]
    ).map({True: "W", False: "L"})
})

home.loc[
    results["home_score"] == results["away_score"],
    "result"
] = "D"

# Partidos como visitante
away = pd.DataFrame({
    "country": results["away_team"],
    "goals_for": results["away_score"],
    "goals_against": results["home_score"],
    "result": (
        results["away_score"] > results["home_score"]
    ).map({True: "W", False: "L"})
})

away.loc[
    results["home_score"] == results["away_score"],
    "result"
] = "D"

# Unir todos los partidos
matches = pd.concat([home, away], ignore_index=True)

# ==========================================================
# Resumen por selección
# ==========================================================

results_summary = (
    matches.groupby("country")
    .agg(
        matches=("country", "count"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum")
    )
    .reset_index()
)

# Victorias, empates y derrotas
results_summary["wins"] = (
    matches[matches["result"] == "W"]
    .groupby("country")
    .size()
    .reindex(results_summary["country"], fill_value=0)
    .values
)

results_summary["draws"] = (
    matches[matches["result"] == "D"]
    .groupby("country")
    .size()
    .reindex(results_summary["country"], fill_value=0)
    .values
)

results_summary["losses"] = (
    matches[matches["result"] == "L"]
    .groupby("country")
    .size()
    .reindex(results_summary["country"], fill_value=0)
    .values
)

# Variables derivadas
results_summary["goal_difference"] = (
    results_summary["goals_for"] -
    results_summary["goals_against"]
)

results_summary["win_rate"] = (
    results_summary["wins"] /
    results_summary["matches"]
)

print("\nResults Summary:", results_summary.shape)
print(results_summary.head())

# ==========================================================
# Conservar únicamente las selecciones presentes en el
# dataset ELO (equipos considerados para el Mundial 2026)
# ==========================================================

valid_countries = set(elo["country"])

results_summary = results_summary[
    results_summary["country"].isin(valid_countries)
].reset_index(drop=True)

players_summary = players_summary[
    players_summary["team_country"].isin(valid_countries)
].reset_index(drop=True)

print("Results Summary:", results_summary.shape)
print("Players Summary:", players_summary.shape)

# ==========================================================
# Normalizar nombres de selecciones nacionales
# ==========================================================

country_mapping = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Curacao": "Curaçao",
    "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde",
    "Czech Republic": "Czechia",
    "Türkiye": "Turkey"
}

ranking["country_full"] = ranking["country_full"].replace(country_mapping)

# ==========================================================
# Comparar selecciones entre datasets
# ==========================================================

elo_countries = set(elo["country"])
players_countries = set(players_summary["team_country"])
results_countries = set(results_summary["country"])

# ==========================================================
# Preparar Ranking (último ranking disponible por selección)
# ==========================================================

ranking_latest = (
    ranking
    .sort_values("rank_date")
    .groupby("country_full", as_index=False)
    .last()
)

print("Ranking Latest:", ranking_latest.shape)


# ==========================================================
# Preparar ELO (último snapshot por selección)
# ==========================================================

elo_latest = (
    elo[elo["snapshot_date"] <= "2026-06-01"]
    .sort_values("snapshot_date")
    .groupby("country", as_index=False)
    .last()
)

print("ELO Latest:", elo_latest.shape)

# ==========================================================
# Construir Master Dataset
# ==========================================================

master = elo_latest.copy()

# ==========================================================
# Integrar datasets
# ==========================================================

# Ranking FIFA
master = master.merge(
    ranking_latest,
    left_on="country",
    right_on="country_full",
    how="left"
).drop(columns=["country_full"])

# Estadísticas de jugadores
master = master.merge(
    players_summary,
    left_on="country",
    right_on="team_country",
    how="left"
).drop(columns=["team_country"])

# Historial de resultados
master = master.merge(
    results_summary,
    on="country",
    how="left",
    suffixes=("_elo", "_history")
)

# ==========================================================
# Guardar dataset maestro
# ==========================================================

master.to_csv(
    DATA_DIR / "master_dataset.csv",
    index=False
)

print("\nMaster Dataset")
print(master.shape)

print("\nMaster Dataset")
print(master.shape)

print("\nNulos restantes:")
print(master.isnull().sum().sum())

print("\nDataset guardado en:")
print(DATA_DIR / "master_dataset.csv")

print("\nMaster Dataset guardado en:")
print(DATA_DIR / "master_dataset.csv")

print("\nValores de wc2026_exit_round:")
print(master["wc2026_exit_round"].value_counts(dropna=False))

world_cups = results[
    results["tournament"] == "FIFA World Cup"
]

print(world_cups["date"].str[:4].value_counts().sort_index())