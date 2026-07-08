# ==========================================================
# FIFA World Cup 2026 Predictor
# Build Training Dataset
#
# Este script construye el dataset de entrenamiento
# combinando los partidos del Mundial 2026 con las
# características de cada selección.
# ==========================================================

from pathlib import Path
import pandas as pd

# ==========================================================
# Definir rutas
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

# ==========================================================
# Cargar datasets
# ==========================================================

master = pd.read_csv(DATA_DIR / "master_dataset.csv")
results = pd.read_csv(DATA_DIR / "results_historical_clean.csv")

print("Master:", master.shape)
print("Results:", results.shape)

# ==========================================================
# Seleccionar únicamente los partidos del Mundial 2026
# ==========================================================

results["date"] = pd.to_datetime(results["date"])

world_cup = results[
    (results["tournament"] == "FIFA World Cup") &
    (results["date"].dt.year == 2026)
].copy()

print("\nPartidos Mundial 2026:", world_cup.shape)

print(world_cup.head())

# ==========================================================
# Agregar características del equipo local
# ==========================================================

home_features = master.add_prefix("home_")

world_cup = world_cup.merge(
    home_features,
    left_on="home_team",
    right_on="home_country",
    how="left"
)

print("\nDespués de unir HOME:")
print(world_cup.shape)

# ==========================================================
# Agregar características del equipo visitante
# ==========================================================

away_features = master.add_prefix("away_")

world_cup = world_cup.merge(
    away_features,
    left_on="away_team",
    right_on="away_country",
    how="left"
)

print("\nDespués de unir AWAY:")
print(world_cup.shape)

print("\nPrimeras columnas:")
print(world_cup.iloc[:5, :15])