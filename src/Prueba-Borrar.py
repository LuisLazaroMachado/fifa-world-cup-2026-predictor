import pandas as pd
df = pd.read_csv("data/processed/historical_training_dataset.csv")
print(df.shape)
print(df.columns.tolist())
print(df['target'].value_counts())