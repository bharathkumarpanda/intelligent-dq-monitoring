import pandas as pd
from dq_engine import check_completeness, check_uniqueness, check_validity, check_consistency

# Load original clean data
df = pd.read_csv("supply_chain_data.csv")

# Inject bad data on purpose
df.loc[5, 'Price'] = -50          # negative price
df.loc[10, 'Price'] = 0           # zero price
df.loc[15, 'Revenue generated'] = 0  # zero revenue
df.loc[20, 'Availability'] = -10  # negative availability
df.loc[25, 'Defect rates'] = 150  # impossible defect rate

# Duplicate a row
df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

# Run DQ checks
completeness = check_completeness(df)
uniqueness = check_uniqueness(df)
validity = check_validity(df)
consistency = check_consistency(df)
overall = round((completeness + uniqueness + validity + consistency) / 4, 2)

print("=" * 40)
print("   DATA QUALITY REPORT (WITH BAD DATA)")
print("=" * 40)
print(f"Completeness  : {completeness}/100")
print(f"Uniqueness    : {uniqueness}/100")
print(f"Validity      : {validity}/100")
print(f"Consistency   : {consistency}/100")
print("-" * 40)
print(f"OVERALL SCORE : {overall}/100")
print("=" * 40)
print("\nProblematic Records Found:")
print(df[df['Price'] <= 0][['SKU', 'Price']])
print(df[df['Defect rates'] > 100][['SKU', 'Defect rates']])
print(df[df['Revenue generated'] <= 0][['SKU', 'Revenue generated']])
