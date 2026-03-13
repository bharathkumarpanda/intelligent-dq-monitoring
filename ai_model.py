import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def find_column(df, name):
    for col in df.columns:
        if col.lower().replace("_", " ") == name.lower():
            return col
    return name

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
        print(anomalies[[sku_col, price_col, defect_col, stock_col, 'status']])

if __name__ == "__main__":
    df = pd.read_csv("supply_chain_data.csv")
    df = detect_anomalies(df)
    show_results(df)
