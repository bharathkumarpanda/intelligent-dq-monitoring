import pandas as pd

def load_data():
    df = pd.read_csv("supply_chain_data.csv")
    return df

def find_column(df, name):
    for col in df.columns:
        if col.lower().replace("_", " ") == name.lower():
            return col
    return name

def check_completeness(df):
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    score = 100 - (missing_cells / total_cells * 100)
    return round(score, 2)

def check_uniqueness(df):
    total_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    score = 100 - (duplicate_rows / total_rows * 100)
    return round(score, 2)

def check_validity(df):
    invalid = 0
    total = len(df)
    price_col = find_column(df, 'Price')
    defect_col = find_column(df, 'Defect Rates')
    avail_col = find_column(df, 'Availability')
    invalid += (pd.to_numeric(df[price_col], errors='coerce') <= 0).sum()
    invalid += (pd.to_numeric(df[defect_col], errors='coerce') > 100).sum()
    invalid += (pd.to_numeric(df[avail_col], errors='coerce') < 0).sum()
    score = 100 - (invalid / total * 100)
    return round(score, 2)

def check_consistency(df):
    inconsistent = 0
    total = len(df)
    rev_col = find_column(df, 'Revenue Generated')
    stock_col = find_column(df, 'Stock Levels')
    inconsistent += (pd.to_numeric(df[rev_col], errors='coerce') <= 0).sum()
    inconsistent += (pd.to_numeric(df[stock_col], errors='coerce') < 0).sum()
    score = 100 - (inconsistent / total * 100)
    return round(score, 2)

def generate_dq_report(df):
    completeness = check_completeness(df)
    uniqueness = check_uniqueness(df)
    validity = check_validity(df)
    consistency = check_consistency(df)
    overall = round((completeness + uniqueness + validity + consistency) / 4, 2)

    print("=" * 40)
    print("     DATA QUALITY REPORT")
    print("=" * 40)
    print(f"Completeness : {completeness}/100")
    print(f"Uniqueness   : {uniqueness}/100")
    print(f"Validity     : {validity}/100")
    print(f"Consistency  : {consistency}/100")
    print("-" * 40)
    print(f"OVERALL SCORE: {overall}/100")
    print("=" * 40)

# Only runs when script is called directly, not when imported
if __name__ == "__main__":
    df = load_data()
    generate_dq_report(df)
