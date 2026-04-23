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
        return df, True  # True = live data
    except Exception:
        df = pd.read_csv("supply_chain_data.csv")
        df.columns = [col.strip().replace("_", " ").title() for col in df.columns]
        return df, False  # False = CSV fallback

st.sidebar.success("✅ Connected to SAP HANA Cloud")
df, is_live = load_data()

# Show data source badge
if is_live:
    st.sidebar.success("📡 Live SAP HANA Data")
else:
    st.sidebar.warning("📂 Using CSV Fallback Data")

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

# ── 1. OVERALL DQ HEALTH ───────────────────────────────────────────────────
st.header("📊 Overall Data Quality Health")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Overall Score", f"{overall}/100")
col2.metric("Completeness", f"{completeness}/100")
col3.metric("Uniqueness", f"{uniqueness}/100")
col4.metric("Validity", f"{validity}/100")
col5.metric("Consistency", f"{consistency}/100")

st.markdown("---")

# ── 2. DQ DIMENSION CHART ──────────────────────────────────────────────────
st.header("📈 DQ Dimension Analysis")
dq_data = pd.DataFrame({
    'Dimension': ['Completeness', 'Uniqueness', 'Validity', 'Consistency'],
    'Score': [completeness, uniqueness, validity, consistency]
})
fig = px.bar(dq_data, x='Dimension', y='Score', color='Score',
             color_continuous_scale='RdYlGn', range_color=[60, 100],
             range_y=[0, 100],
             text='Score')
fig.update_traces(texttemplate='%{text}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── 3. AI ANOMALY DETECTION ────────────────────────────────────────────────
st.header("🤖 AI Anomaly Detection")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Normal Records", len(df) - len(anomalies))
col3.metric("🚨 Suspicious Records", len(anomalies))

if len(anomalies) > 0:
    st.subheader("🚨 Suspicious Records Found By AI")
    display_cols = [c for c in [
        'Sku', 'Product Type', 'Price', 'Defect Rates',
        'Stock Levels', 'status', 'anomaly_reason'
    ] if c in df.columns]
    st.dataframe(
        anomalies[display_cols].rename(columns={'anomaly_reason': '⚠️ Why Flagged?'}),
        use_container_width=True
    )

st.markdown("---")

# ── 4. BUSINESS IMPACT CARD ────────────────────────────────────────────────
st.header("💼 Business Impact Assessment")

num_anomalies = len(anomalies)
avg_revenue = df['Revenue Generated'].mean() if 'Revenue Generated' in df.columns else 5000
estimated_risk = round(num_anomalies * avg_revenue * 0.15, 2)  # 15% revenue at risk per anomaly
detection_rate = round((num_anomalies / len(df)) * 100, 1)

col1, col2, col3 = st.columns(3)
col1.metric(
    "🚨 Anomalies Detected",
    num_anomalies,
    help="Records flagged by Isolation Forest ML model"
)
col2.metric(
    "💰 Estimated Revenue at Risk",
    f"₹{estimated_risk:,.0f}",
    help="15% of avg revenue × number of suspicious records"
)
col3.metric(
    "📊 Anomaly Detection Rate",
    f"{detection_rate}%",
    help="Percentage of records flagged as suspicious"
)

st.info(
    f"🛡️ **Proactive Alert:** {num_anomalies} supply chain records show unusual patterns. "
    f"Early detection prevents potential reporting errors and financial discrepancies "
    f"worth approximately ₹{estimated_risk:,.0f} in revenue impact."
)

st.markdown("---")

# ── 5. DATA SOURCE ─────────────────────────────────────────────────────────
st.header("🔗 Data Source")
st.info("📡 Live data from SAP HANA Cloud | Instance: dq-monitoring-hana | Region: Singapore - Azure")

# ── 6. COMPLETE DATASET ────────────────────────────────────────────────────
st.header("📋 Complete Dataset View")

# Filter controls
with st.expander("🔽 Filter Dataset"):
    if 'Product Type' in df.columns:
        product_types = ['All'] + sorted(df['Product Type'].dropna().unique().tolist())
        selected_type = st.selectbox("Filter by Product Type", product_types)
        if selected_type != 'All':
            df_display = df[df['Product Type'] == selected_type]
        else:
            df_display = df
    else:
        df_display = df

    show_suspicious_only = st.checkbox("Show Suspicious Records Only")
    if show_suspicious_only:
        df_display = df_display[df_display['status'] == 'SUSPICIOUS']

st.dataframe(df_display, use_container_width=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info("""
**Intelligent DQ Monitoring System**
- Built with SAP HANA Cloud + AI
- Isolation Forest Anomaly Detection
- Real-time Data Quality Scoring
- Business Impact Assessment
- B.Tech Special Project 2026
- By: Bharath Kumar Panda
""")

import datetime
st.sidebar.markdown(f"🕒 Last refreshed: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}")

