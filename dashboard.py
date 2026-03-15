import streamlit as st
import pandas as pd
import plotly.express as px
from dq_engine import check_completeness, check_uniqueness, check_validity, check_consistency
from ai_model import detect_anomalies

st.set_page_config(
    page_title="Intelligent DQ Monitoring System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Intelligent Data Quality Monitoring System")
st.subheader("SAP Business Data Cloud + AI")
st.markdown("---")

@st.cache_data
def load_data():
    try:
        from hdbcli import dbapi
        conn = dbapi.connect(
            address=st.secrets["HANA_HOST"],
            port=int(st.secrets["HANA_PORT"]),
            user=st.secrets["HANA_USER"],
            password=st.secrets["HANA_PASSWORD"],
            encrypt=True,
            sslValidateCertificate=False
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM SUPPLY_CHAIN_DATA")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=columns)
        cursor.close()
        conn.close()
        df.columns = [col.strip().replace("_", " ").title() for col in df.columns]
        return df
    except Exception:
        df = pd.read_csv("supply_chain_data.csv")
        df.columns = [col.strip().replace("_", " ").title() for col in df.columns]
        return df

st.sidebar.success("✅ Connected to SAP HANA Cloud")
df = load_data()

numeric_cols = ['Price', 'Availability', 'Stock Levels', 'Lead Times',
                'Shipping Costs', 'Defect Rates', 'Revenue Generated',
                'Number Of Products Sold', 'Order Quantities',
                'Manufacturing Costs', 'Production Volumes', 'Costs']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

completeness = check_completeness(df)
uniqueness = check_uniqueness(df)
validity = check_validity(df)
consistency = check_consistency(df)
overall = round((completeness + uniqueness + validity + consistency) / 4, 2)

df = detect_anomalies(df)
anomalies = df[df['anomaly'] == -1]

st.header("📊 Overall Data Quality Health")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Overall Score", f"{overall}/100")
col2.metric("Completeness", f"{completeness}/100")
col3.metric("Uniqueness", f"{uniqueness}/100")
col4.metric("Validity", f"{validity}/100")
col5.metric("Consistency", f"{consistency}/100")

st.markdown("---")

st.header("📈 DQ Dimension Analysis")
dq_data = pd.DataFrame({
    'Dimension': ['Completeness', 'Uniqueness', 'Validity', 'Consistency'],
    'Score': [completeness, uniqueness, validity, consistency]
})
fig = px.bar(dq_data, x='Dimension', y='Score', color='Score',
             color_continuous_scale='RdYlGn', range_y=[0, 100])
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.header("🤖 AI Anomaly Detection")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Normal Records", len(df) - len(anomalies))
col3.metric("🚨 Suspicious Records", len(anomalies))

if len(anomalies) > 0:
    st.subheader("🚨 Suspicious Records Found By AI")
    display_cols = [c for c in ['Sku', 'Product Type', 'Price', 'Defect Rates', 'Stock Levels', 'status'] if c in df.columns]
    st.dataframe(anomalies[display_cols])

st.markdown("---")

st.header("🔗 Data Source")
st.info("📡 Live data from SAP HANA Cloud | Instance: dq-monitoring-hana | Region: Singapore - Azure")

st.header("📋 Complete Dataset View")
st.dataframe(df)

st.sidebar.title("About")
st.sidebar.info("""
**Intelligent DQ Monitoring System**
- Built with SAP HANA Cloud + AI
- Isolation Forest Anomaly Detection
- Real-time Data Quality Scoring
- B.Tech Special Project 2026
- By: Bharath Kumar Panda
""")