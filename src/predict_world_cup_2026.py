from pathlib import Path
import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

model = joblib.load(DATA_DIR / "mlp_model.pkl")

matches = pd.read_csv(DATA_DIR / "world_cup_2026_features.csv")

print(matches.columns.tolist())
print(matches.head())

feature_names = model.named_steps["prep"].feature_names_in_

# Obtener exactamente las columnas con las que fue entrenado el modelo
feature_names = model.named_steps["prep"].feature_names_in_

X = matches.reindex(columns=feature_names)

pred = model.predict(X)

resultado = matches[["home_team", "away_team"]].copy()

resultado["prediction"] = pred

resultado["winner"] = resultado.apply(
    lambda r: r["home_team"] if r["prediction"] == 1
    else (
        r["away_team"] if r["prediction"] == 0
        else "Draw"
    ),
    axis=1,
)

print(resultado)

resultado.to_csv(
    DATA_DIR / "predictions_2026.csv",
    index=False
)

print("\nGuardado en:")
print(DATA_DIR / "predictions_2026.csv")

matches = pd.read_csv(DATA_DIR / "world_cup_2026_features.csv")

print(matches.columns.tolist())
print(matches.head())