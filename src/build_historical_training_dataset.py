from pathlib import Path
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# ==========================
# Cargar datasets
# ==========================

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

# ==========================================================
# ELO: último snapshot anterior a la fecha del partido
# ==========================================================

world_cup = world_cup.sort_values("date").reset_index(drop=True)
elo_sorted = elo.sort_values("snapshot_date")

for side in ["home", "away"]:
    world_cup = pd.merge_asof(
        world_cup,
        elo_sorted.add_prefix(f"{side}_"),
        left_on="date",
        right_on=f"{side}_snapshot_date",
        left_by=f"{side}_team",
        right_by=f"{side}_country",
        direction="backward",
        allow_exact_matches=False,
    )

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

world_cup["rank_diff"] = world_cup["away_rank"] - world_cup["home_rank"]

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

for side in ["home", "away"]:
    print(side, "snapshot posterior al partido:",
          (world_cup[f"{side}_snapshot_date"] > world_cup["date"]).mean())
print(world_cup[["home_rating", "away_rating"]].isna().sum())