from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Cargar datos
df = pd.read_csv(DATA / "historical_training_dataset.csv")

drop_cols = ["date","home_team","away_team","winner","target",
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
             "tournament","city","country","neutral",
             "home_rank_date","away_rank_date"]

X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include="number")
y = df["target"].values

# Preprocesamiento
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()
X = imputer.fit_transform(X)
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Tensores
X_tr = torch.tensor(X_train, dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.long)
X_te = torch.tensor(X_test,  dtype=torch.float32)
y_te = torch.tensor(y_test,  dtype=torch.long)

loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)

# Arquitectura MLP
n_features = X_train.shape[1]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),         nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),          nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16),          nn.ReLU(),
            nn.Linear(16, 3)
        )
    def forward(self, x):
        return self.net(x)

model_adam = MLP()
model_sgd  = MLP()

criterion = nn.CrossEntropyLoss()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.001)
optimizer_sgd  = torch.optim.SGD(model_sgd.parameters(),  lr=0.01, momentum=0.9)

# Entrenamiento
def train(model, optimizer, epochs=50):
    losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/50 - Loss: {losses[-1]:.4f}")
    return losses

print("Entrenando MLP con Adam...")
losses_adam = train(model_adam, optimizer_adam)

print("\nEntrenando MLP con SGD + Momentum...")
losses_sgd = train(model_sgd, optimizer_sgd)

# Evaluación
def evaluate(model, name):
    model.eval()
    with torch.no_grad():
        preds = model(X_te).argmax(dim=1).numpy()
    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds,
          target_names=["Away Win","Home Win","Draw"]))
    acc = (preds == y_test).mean()
    print(f"Accuracy: {acc:.4f}")
    return acc

acc_adam = evaluate(model_adam, "MLP Adam")
acc_sgd  = evaluate(model_sgd,  "MLP SGD+Momentum")

# Guardar mejor modelo
best = model_adam if acc_adam >= acc_sgd else model_sgd
torch.save(best.state_dict(), MODELS / "mlp_final.pt")
joblib.dump({"imputer": imputer, "scaler": scaler, "n_features": n_features},
            DATA / "mlp_preprocessor.pkl")

print("\n✓ MLP guardado en models/mlp_final.pt")
print(f"  Adam accuracy:        {acc_adam:.4f}")
print(f"  SGD+Momentum accuracy:{acc_sgd:.4f}")

# Guardar losses para graficar
pd.DataFrame({
    "epoch": range(1, 51),
    "loss_adam": losses_adam,
    "loss_sgd":  losses_sgd
}).to_csv(DATA / "mlp_training_losses.csv", index=False)
print("✓ Curvas de loss guardadas")