import pandas as pd

matches = pd.read_csv(
    "data/processed/group_stage_matches.csv"
)

print(matches.head(20))

print()

print(matches.groupby("group").size())