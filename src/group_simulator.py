from itertools import combinations
import pandas as pd

ROOT = "data/processed"

groups = pd.read_csv(f"{ROOT}/world_cup_groups.csv")

matches = []

for group in sorted(groups["group"].unique()):

    teams = (
        groups.loc[
            groups["group"] == group,
            "team"
        ].tolist()
    )

    for home, away in combinations(teams, 2):

        matches.append({
            "group": group,
            "home_team": home,
            "away_team": away
        })

matches = pd.DataFrame(matches)

print(matches)

print()
print("Total partidos:", len(matches))

matches.to_csv(
    f"{ROOT}/group_stage_matches.csv",
    index=False
)

print("\nGuardado.")