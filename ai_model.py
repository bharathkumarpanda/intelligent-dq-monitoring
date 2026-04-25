import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def find_column(df, name):
    for col in df.columns:
        if col.lower().replace("_", " ").strip() == name.lower().replace("_", " ").strip():
            return col
    return None

def get_anomaly_reason(row, df, features):
    reasons = []
    price_col = find_column(df, 'Price')
    defect_col = find_column(df, 'Defect Rates')
    stock_col = find_column(df, 'Stock Levels')
    lead_col = find_column(df, 'Lead Times')
    ship_col = find_column(df, 'Shipping Costs')
    avail_col = find_column(df, 'Availability')

    if price_col:
        if row[price_col] > df[price_col].quantile(0.80):
            reasons.append(f"High price ({row[price_col]:.1f}, top 20%)")
        elif row[price_col] < df[price_col].quantile(0.20):
            reasons.append(f"Low price ({row[price_col]:.1f}, bottom 20%)")

    if defect_col:
        if row[defect_col] > df[defect_col].quantile(0.75):
            reasons.append(f"High defect rate ({row[defect_col]:.2f}%)")

    if stock_col:
        if row[stock_col] < df[stock_col].quantile(0.25):
            reasons.append(f"Low stock ({int(row[stock_col])} units)")
        elif row[stock_col] > df[stock_col].quantile(0.75):
            reasons.append(f"High stock ({int(row[stock_col])} units)")

    if avail_col:
        if row[avail_col] < df[avail_col].quantile(0.20):
            reasons.append(f"Low availability ({row[avail_col]:.0f}%)")

    if ship_col:
        if row[ship_col] > df[ship_col].quantile(0.75):
            reasons.append(f"High shipping cost ({row[ship_col]:.1f})")

    if not reasons:
        reasons.append("Anomalous pattern across multiple features (ML detected)")

    return " | ".join(reasons)


def detect_anomalies(df):
    feature_names = ['Price', 'Availability', 'Stock Levels',
                     'Lead Times', 'Shipping Costs', 'Defect Rates']
    features = [find_column(df, f) for f in feature_names]
    features = [f for f in features if f is not None]

    X = df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(X)
    df['risk_score'] = model.decision_function(X)
    df['status'] = df['anomaly'].apply(
        lambda x: 'SUSPICIOUS' if x == -1 else 'NORMAL'
    )
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
        cols = [c for c in [sku_col, price_col, defect_col, stock_col, 'status', 'anomaly_reason'] if c]
        print(anomalies[cols])

if __name__ == "__main__":
    df = pd.read_csv("supply_chain_data.csv")
    df = detect_anomalies(df)
    show_results(df)
