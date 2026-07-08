from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# ==========================
# Cargar datasets
# ==========================

ranking = pd.read_csv(
    DATA_DIR / "fifa_world_ranking_clean.csv",
    parse_dates=["rank_date"]
)

results = pd.read_csv(
    DATA_DIR / "results_historical_clean.csv",
    parse_dates=["date"]
)

elo = pd.read_csv(
    DATA_DIR / "elo_ratings_wc2026_clean.csv",
    parse_dates=["snapshot_date"]
)

# ==========================
# Solo partidos de Mundial
# ==========================

world_cup = results[
    results["tournament"] == "FIFA World Cup"
].copy()

world_cup["date"] = pd.to_datetime(world_cup["date"])

world_cup["year"] = world_cup["date"].dt.year

print(world_cup.shape)
print(world_cup["year"].value_counts().sort_index())

# ==========================
# Crear target
# ==========================

world_cup["target"] = 2

world_cup.loc[
    world_cup["home_score"] > world_cup["away_score"],
    "target"
] = 1

world_cup.loc[
    world_cup["home_score"] < world_cup["away_score"],
    "target"
] = 0

print(world_cup[["year","home_team","away_team","target"]].head())

world_cup = world_cup[
    (world_cup["year"] >= 1994) &
    (world_cup["year"] < 2026)
].reset_index(drop=True)

print(world_cup.shape)
print(world_cup["year"].value_counts().sort_index())

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

print()

print("Ranking:", ranking.shape)
print("ELO:", elo.shape)

print()

print(ranking.columns.tolist())
print(elo.columns.tolist())


# ==========================================================
# Ranking FIFA del mismo año del Mundial
# ==========================================================

ranking["year"] = ranking["rank_date"].dt.year

ranking_latest = (
    ranking
    .sort_values("rank_date")
    .groupby(["country_full", "year"], as_index=False)
    .last()
)

print(ranking_latest.shape)

world_cup = world_cup.merge(
    ranking_latest.add_prefix("home_"),
    left_on=["home_team", "year"],
    right_on=["home_country_full", "home_year"],
    how="left"
)

print("Después HOME:", world_cup.shape)

world_cup = world_cup.merge(
    ranking_latest.add_prefix("away_"),
    left_on=["away_team", "year"],
    right_on=["away_country_full", "away_year"],
    how="left"
)

print("Después AWAY:", world_cup.shape)

print(world_cup.head())


# ==========================================================
# ELO del mismo año
# ==========================================================

elo_latest = (
    elo
    .sort_values("snapshot_date")
    .groupby(["country", "year"], as_index=False)
    .last()
)

print(elo_latest.shape)

world_cup = world_cup.merge(
    elo_latest.add_prefix("home_"),
    left_on=["home_team", "year"],
    right_on=["home_country", "home_year"],
    how="left"
)

print("Después HOME ELO:", world_cup.shape)

world_cup = world_cup.merge(
    elo_latest.add_prefix("away_"),
    left_on=["away_team", "year"],
    right_on=["away_country", "away_year"],
    how="left"
)

print("Después AWAY ELO:", world_cup.shape)

print(world_cup.head())

# ==========================================================
# Limpiar columnas auxiliares
# ==========================================================

drop_columns = [
    "home_country_full",
    "home_year",
    "away_country_full",
    "away_year",
    "home_country",
    "away_country"
]

world_cup = world_cup.drop(
    columns=[c for c in drop_columns if c in world_cup.columns]
)

# ==========================================
# Variables diferenciales
# ==========================================

world_cup["rank_diff"] = (
    world_cup["away_rank_x"] -
    world_cup["home_rank_x"]
)

world_cup["points_diff"] = (
    world_cup["home_total_points"] -
    world_cup["away_total_points"]
)

world_cup["rank_change_diff"] = (
    world_cup["home_rank_change"] -
    world_cup["away_rank_change"]
)

world_cup["elo_diff"] = (
    world_cup["home_rating"] -
    world_cup["away_rating"]
)

world_cup["win_rate_diff"] = (
    world_cup["home_wins"] /
    (world_cup["home_matches_total"] + 1)
) - (
    world_cup["away_wins"] /
    (world_cup["away_matches_total"] + 1)
)

world_cup["goal_diff_diff"] = (
    (world_cup["home_goals_for"] - world_cup["home_goals_against"]) -
    (world_cup["away_goals_for"] - world_cup["away_goals_against"])
)

world_cup["host_advantage"] = (
    world_cup["home_is_host"].fillna(0) -
    world_cup["away_is_host"].fillna(0)
)

print("\nDataset final:", world_cup.shape)

print("\nNulos:")
print(world_cup.isnull().sum().sort_values(ascending=False).head(30))

# ==========================================================
# Guardar dataset
# ==========================================================

world_cup.to_csv(
    DATA_DIR / "historical_training_dataset.csv",
    index=False
)

print("\nDataset guardado en:")
print(DATA_DIR / "historical_training_dataset.csv")

print("\nEquipos HOME sin ELO:")
print(
    sorted(
        world_cup.loc[
            world_cup["home_rating"].isna(),
            "home_team"
        ].unique()
    )
)

print("\nEquipos AWAY sin ELO:")
print(
    sorted(
        world_cup.loc[
            world_cup["away_rating"].isna(),
            "away_team"
        ].unique()
    )
)

print("\nEquipos HOME sin Ranking:")
print(
    sorted(
        world_cup.loc[
            world_cup["home_total_points"].isna(),
            "home_team"
        ].unique()
    )
)

print("\nEquipos AWAY sin Ranking:")
print(
    sorted(
        world_cup.loc[
            world_cup["away_total_points"].isna(),
            "away_team"
        ].unique()
    )
)