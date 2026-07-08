from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"

master = pd.read_csv(DATA_DIR / "master_dataset.csv")

print(master["wc2026_exit_round"].value_counts())