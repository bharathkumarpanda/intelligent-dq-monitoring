import pandas as pd

df = pd.read_csv("supply_chain_data.csv")

print("Dataset loaded successfully!")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("\nFirst 5 rows:")
print(df.head())