from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

df = pd.read_csv(DATA_DIR / "training_matches.csv", parse_dates=["date"])
df["elo_diff_n"] = df["elo_diff"] / 400

BASE = ["elo_diff_n", "neutral"]
FORM = BASE + ["form_pts_diff", "form_gd_diff"]
LABELS = [0, 1, 2]

wc = df["tournament"] == "FIFA World Cup"
EXAMS = {
    "Mundial 2018": df[wc & (df["date"].dt.year == 2018)],
    "Mundial 2022": df[wc & (df["date"].dt.year == 2022)],
    "Competitivos 2022 (ene-nov, sin amistosos)": df[
        (df["date"] >= "2022-01-01") & (df["date"] < "2022-11-20")
        & (df["tournament"] != "Friendly")
    ],
}


def evaluate(name, test):
    train = df[df["date"] < test["date"].min()]   # solo el pasado del examen
    y = test["target"]
    rows = []

    majority = train["target"].mode()[0]
    rows.append(("Siempre la clase más frecuente",
                 accuracy_score(y, [majority] * len(test)), None))
    rows.append(("Gana el de mayor Elo",
                 accuracy_score(y, (test["elo_diff"] > 0).astype(int)), None))

    prior = DummyClassifier(strategy="prior").fit(train[BASE], train["target"])
    rows.append(("Solo frecuencias históricas", None,
                 log_loss(y, prior.predict_proba(test[BASE]), labels=LABELS)))

    models = {
        "Logística (elo + neutral)": (LogisticRegression(max_iter=1000), BASE),
        "Logística (+ forma reciente)": (LogisticRegression(max_iter=1000), FORM),
        "Gradient Boosting (+ forma)": (
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=200, random_state=42), FORM),
    }
    for label, (model, feats) in models.items():
        model.fit(train[feats], train["target"])
        rows.append((label,
                     accuracy_score(y, model.predict(test[feats])),
                     log_loss(y, model.predict_proba(test[feats]), labels=LABELS)))

    out = pd.DataFrame(rows, columns=["modelo", "accuracy", "log_loss"])
    print(f"\n=== {name} | train: {len(train)} | test: {len(test)} partidos ===")
    print(out.round(3).to_string(index=False))


for name, test in EXAMS.items():
    evaluate(name, test)