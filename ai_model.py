import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def find_column(df, name):
    for col in df.columns:
        if col.lower().replace("_", " ") == name.lower():
            return col
    return name

def get_anomaly_reason(row, df, features):
    """Generate a human-readable reason why a record was flagged."""
    reasons = []

    price_col = find_column(df, 'Price')
    defect_col = find_column(df, 'Defect Rates')
    stock_col = find_column(df, 'Stock Levels')
    lead_col = find_column(df, 'Lead Times')
    ship_col = find_column(df, 'Shipping Costs')

    if price_col in df.columns:
        mean_price = df[price_col].mean()
        std_price = df[price_col].std()
        if row[price_col] > mean_price + 2 * std_price:
            reasons.append(f"Price outlier (high: {row[price_col]:.1f})")
        elif row[price_col] < mean_price - 2 * std_price:
            reasons.append(f"Price outlier (low: {row[price_col]:.1f})")

    if defect_col in df.columns:
        mean_defect = df[defect_col].mean()
        std_defect = df[defect_col].std()
        if row[defect_col] > mean_defect + 2 * std_defect:
            reasons.append(f"High defect rate ({row[defect_col]:.2f}%)")

    if stock_col in df.columns:
        mean_stock = df[stock_col].mean()
        std_stock = df[stock_col].std()
        if row[stock_col] < mean_stock - 2 * std_stock:
            reasons.append(f"Critically low stock ({int(row[stock_col])} units)")
        elif row[stock_col] > mean_stock + 2 * std_stock:
            reasons.append(f"Unusually high stock ({int(row[stock_col])} units)")

    if lead_col in df.columns:
        mean_lead = df[lead_col].mean()
        std_lead = df[lead_col].std()
        if row[lead_col] > mean_lead + 2 * std_lead:
            reasons.append(f"Long lead time ({row[lead_col]:.0f} days)")

    if ship_col in df.columns:
        mean_ship = df[ship_col].mean()
        std_ship = df[ship_col].std()
        if row[ship_col] > mean_ship + 2 * std_ship:
            reasons.append(f"High shipping cost ({row[ship_col]:.1f})")

    # Combination flag: high defect + low stock = supply risk
    if defect_col in df.columns and stock_col in df.columns:
        if row[defect_col] > df[defect_col].mean() and row[stock_col] < df[stock_col].mean():
            if not any("defect" in r.lower() for r in reasons):
                reasons.append("High defect + low stock (supply risk)")

    if not reasons:
        reasons.append("Unusual combination of multiple features")

    return " | ".join(reasons)


def detect_anomalies(df):
    feature_names = ['Price', 'Availability', 'Stock Levels',
                     'Lead Times', 'Shipping Costs', 'Defect Rates']
    features = [find_column(df, f) for f in feature_names]

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    df['risk_score'] = model.decision_function(X)
    df['status'] = df['anomaly'].apply(
        lambda x: 'SUSPICIOUS' if x == -1 else 'NORMAL'
    )

    # Add reason column
    df['anomaly_reason'] = df.apply(
        lambda row: get_anomaly_reason(row, df, features) if row['anomaly'] == -1 else '-',
        axis=1
    )

    return df


def show_results(df):
    anomalies = df[df['anomaly'] == -1]
    normal = df[df['anomaly'] == 1]
    print("=" * 50)
    print("     AI ANOMALY DETECTION REPORT")
    print("=" * 50)
    print(f"Total Records Analyzed : {len(df)}")
    print(f"Normal Records         : {len(normal)}")
    print(f"Suspicious Records     : {len(anomalies)}")
    print("=" * 50)
    if len(anomalies) > 0:
        print("\nSUSPICIOUS RECORDS FOUND BY AI:")
        sku_col = find_column(df, 'Sku')
        price_col = find_column(df, 'Price')
        defect_col = find_column(df, 'Defect Rates')
        stock_col = find_column(df, 'Stock Levels')
        print(anomalies[[sku_col, price_col, defect_col, stock_col, 'status', 'anomaly_reason']])

if __name__ == "__main__":
    df = pd.read_csv("supply_chain_data.csv")
    df = detect_anomalies(df)
    show_results(df)