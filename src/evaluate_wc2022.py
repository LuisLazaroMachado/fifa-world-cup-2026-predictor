import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from sklearn.impute import SimpleImputer

DATA = Path("data/processed")

df = pd.read_csv(DATA / "historical_training_dataset.csv")

drop_cols = [
    "date","home_team","away_team","winner","target",
    "home_score","away_score","went_to_shootout","year",
    "home_year_x","away_year_x","home_year_y","away_year_y",
    "home_rank_x","away_rank_x","home_total_points","away_total_points",
    "home_rank_change","away_rank_change","home_rating","away_rating",
    "home_wc2026_exit_round","away_wc2026_exit_round",
    "home_snapshot_date","away_snapshot_date",
    "home_rank_date","away_rank_date",
    "home_country_abrv","away_country_abrv",
    "home_country_code","away_country_code",
    "home_confederation_x","away_confederation_x",
    "home_confederation_y","away_confederation_y",
    "tournament","city","country","neutral"
]

# Test set = Mundial 2022 (datos reales)
test  = df[df["date"].str.contains("2022")]
train = df[~df.index.isin(test.index)]

print(f"Train: {len(train)} partidos")
print(f"Test (Mundial 2022): {len(test)} partidos")

X_train = train.drop(columns=drop_cols, errors="ignore").select_dtypes(include="number")
X_test  = test.drop(columns=drop_cols,  errors="ignore").select_dtypes(include="number")
y_train = train["target"]
y_test  = test["target"]

imp = SimpleImputer(strategy="median")
X_train = imp.fit_transform(X_train)
X_test  = imp.transform(X_test)

model = joblib.load(DATA / "rf_model.pkl")
pred  = model.predict(X_test)

print("\n=== Random Forest - Test Set Mundial 2022 ===")
print(classification_report(
    y_test, pred,
    target_names=["Away Win","Home Win","Draw"]
))
f1 = f1_score(y_test, pred, average="weighted")
print(f"F1 weighted: {f1:.3f}")