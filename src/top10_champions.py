from collections import Counter
import pandas as pd
from tqdm import trange

from simulate_knockout import simulate_world_cup

N = 500

champions = Counter()

print(f"\nEjecutando {N} simulaciones...\n")

for _ in trange(N):

    champion = simulate_world_cup(False)

    champions[champion] += 1

ranking = (
    pd.DataFrame(
        champions.items(),
        columns=["team", "titles"]
    )
    .sort_values(
        "titles",
        ascending=False
    )
)

ranking["probability"] = ranking["titles"] / N

ranking.to_csv(
    "data/processed/top10_champions.csv",
    index=False
)

print("\nTOP 10 CAMPEONES\n")
print(ranking.head(10))