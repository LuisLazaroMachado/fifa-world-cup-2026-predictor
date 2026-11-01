import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

qualified = pd.read_csv(DATA / "qualified_teams.csv")

qualified = qualified["team"].tolist()

print("Clasificados:", len(qualified))
print()

for i, team in enumerate(qualified, 1):
    print(i, team)