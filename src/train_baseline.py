from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

df = pd.read_csv(DATA_DIR / "training_matches.csv", parse_dates=["date"])
df["elo_diff_n"] = df["elo_diff"] / 400
FEATURES = ["elo_diff_n", "neutral"]
LABELS = [0, 1, 2]


def evaluate(year):
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == year)]
    train = df[df["date"] < wc["date"].min()]   # solo el pasado del Mundial evaluado
    y_test = wc["target"]
    rows = []

    majority = train["target"].mode()[0]
    rows.append(("Siempre la clase más frecuente",
                 accuracy_score(y_test, [majority] * len(wc)), None))

    higher_elo = (wc["elo_diff"] > 0).astype(int)   # 1 = gana local, 0 = visitante
    rows.append(("Gana el de mayor Elo",
                 accuracy_score(y_test, higher_elo), None))

    prior = DummyClassifier(strategy="prior").fit(train[FEATURES], train["target"])
    rows.append(("Solo frecuencias históricas",
                 None, log_loss(y_test, prior.predict_proba(wc[FEATURES]), labels=LABELS)))

    lr = LogisticRegression(max_iter=1000).fit(train[FEATURES], train["target"])
    proba = lr.predict_proba(wc[FEATURES])
    rows.append(("Regresión logística (elo_diff + neutral)",
                 accuracy_score(y_test, lr.predict(wc[FEATURES])),
                 log_loss(y_test, proba, labels=LABELS)))

    out = pd.DataFrame(rows, columns=["modelo", "accuracy", "log_loss"])
    print(f"\n=== Mundial {year} | train: {len(train)} partidos | test: {len(wc)} partidos ===")
    print(out.round(3).to_string(index=False))


for year in (2018, 2022):
    evaluate(year)