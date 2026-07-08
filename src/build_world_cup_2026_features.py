from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# ==========================
# Cargar datasets
# ==========================

matches = pd.read_csv(DATA_DIR / "master_dataset.csv")

ranking = pd.read_csv(
    DATA_DIR / "fifa_world_ranking_clean.csv",
    parse_dates=["rank_date"]
)

elo = pd.read_csv(
    DATA_DIR / "elo_ratings_wc2026_clean.csv",
    parse_dates=["snapshot_date"]
)

# ==========================
# Año Mundial
# ==========================

matches["year"] = 2026

# ==========================
# Unificar nombres
# ==========================

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
elo["country"] = elo["country"].replace(country_mapping)

# ==========================
# Ranking FIFA
# ==========================

ranking["year"] = ranking["rank_date"].dt.year

ranking_latest = (
    ranking
    .sort_values("rank_date")
    .groupby(["country_full", "year"], as_index=False)
    .last()
)

matches = matches.merge(
    ranking_latest.add_prefix("home_"),
    left_on=["home_team", "year"],
    right_on=["home_country_full", "home_year"],
    how="left"
)

matches = matches.merge(
    ranking_latest.add_prefix("away_"),
    left_on=["away_team", "year"],
    right_on=["away_country_full", "away_year"],
    how="left"
)

# ==========================
# ELO
# ==========================

elo_latest = (
    elo
    .sort_values("snapshot_date")
    .groupby(["country", "year"], as_index=False)
    .last()
)

matches = matches.merge(
    elo_latest.add_prefix("home_"),
    left_on=["home_team", "year"],
    right_on=["home_country", "home_year"],
    how="left"
)

matches = matches.merge(
    elo_latest.add_prefix("away_"),
    left_on=["away_team", "year"],
    right_on=["away_country", "away_year"],
    how="left"
)

# ==========================
# Variables diferenciales
# ==========================

matches["rank_diff"] = (
    matches["away_rank_x"] -
    matches["home_rank_x"]
)

matches["points_diff"] = (
    matches["home_total_points"] -
    matches["away_total_points"]
)

matches["rank_change_diff"] = (
    matches["home_rank_change"] -
    matches["away_rank_change"]
)

matches["elo_diff"] = (
    matches["home_rating"] -
    matches["away_rating"]
)

matches["win_rate_diff"] = (
    matches["home_wins"] /
    (matches["home_matches_total"] + 1)
) - (
    matches["away_wins"] /
    (matches["away_matches_total"] + 1)
)

matches["goal_diff_diff"] = (
    (matches["home_goals_for"] - matches["home_goals_against"])
    -
    (matches["away_goals_for"] - matches["away_goals_against"])
)

matches["host_advantage"] = (
    matches["home_is_host"].fillna(0)
    -
    matches["away_is_host"].fillna(0)
)

# ==========================
# Eliminar columnas auxiliares
# ==========================

drop_cols = [
    "home_country_full",
    "away_country_full",
    "home_country",
    "away_country"
]

matches = matches.drop(
    columns=[c for c in drop_cols if c in matches.columns],
    errors="ignore"
)

# ==========================
# Guardar
# ==========================

matches.to_csv(
    DATA_DIR / "world_cup_2026_features.csv",
    index=False
)

print(matches.shape)
print(matches[["home_team", "away_team"]].head())
print(matches.isnull().sum().sort_values(ascending=False).head(20))