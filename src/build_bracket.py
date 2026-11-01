from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"

teams = pd.read_csv(DATA / "qualified_teams.csv")["team"].tolist()

matches = [
    [teams[0], teams[23]],
    [teams[1], teams[22]],
    [teams[2], teams[21]],
    [teams[3], teams[20]],
    [teams[4], teams[19]],
    [teams[5], teams[18]],
    [teams[6], teams[17]],
    [teams[7], teams[16]],
    [teams[8], teams[15]],
    [teams[9], teams[14]],
    [teams[10], teams[13]],
    [teams[11], teams[12]],
]

bracket = pd.DataFrame(
    matches,
    columns=["home_team", "away_team"]
)

print(bracket)

bracket.to_csv(
    DATA / "round_of_32.csv",
    index=False
)

print("\nGuardado en:")
print(DATA / "round_of_32.csv")