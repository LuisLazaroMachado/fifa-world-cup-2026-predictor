from pathlib import Path

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

df = pd.read_csv(DATA_DIR / "historical_training_dataset.csv")

print(df.shape)

# Eliminar columnas que no son features
drop_cols = [
    "date",
    "home_team",
    "away_team",
    "winner",
    "target",
    "home_score",
    "away_score"
]

X = df.drop(columns=drop_cols)

# Eliminar variables que no existirían antes del partido
X = X.drop(columns=[
    "went_to_shootout",
    "year",
    "home_year_x",
    "away_year_x",
    "home_year_y",
    "away_year_y",
    "home_rank_x",
    "away_rank_x",
    "home_total_points",
    "away_total_points",
    "home_rank_change",
    "away_rank_change",
    "home_rating",
    "away_rating"
], errors="ignore")

y = df["target"]

X = X.select_dtypes(include="number")

print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)

joblib.dump(
    model,
    DATA_DIR / "rf_model.pkl"
)

print("Modelo Random Forest guardado.")

print("\nNúmero de features:", len(X.columns))
print("\nFeatures usadas:")
print(X.columns.tolist())

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

joblib.dump(model, DATA_DIR / "mlp_model.pkl")

print("Modelo guardado en:")
print(DATA_DIR / "mlp_model.pkl")

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "mlp_model.pkl"
)

print("\nModelo guardado.")

pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))