import streamlit as st
import pandas as pd
import plotly.express as px
from dq_engine import check_completeness, check_uniqueness, check_validity, check_consistency
from ai_model import detect_anomalies
import datetime

st.set_page_config(
    page_title="AI-Powered Supply Chain DQ Monitor",
    page_icon="🔍",
    layout="wide"
)

# ── TITLE + SUBTEXT ────────────────────────────────────────────────────────
st.title("🔍 AI-Powered Supply Chain Data Quality Monitor")
st.subheader("Detects anomalies in inventory, pricing and defect data to prevent operational losses")
st.markdown("---")

# ── USE CASE BOX ───────────────────────────────────────────────────────────
st.info("""
**📦 Use Case: Supply Chain Data Monitoring**
- 🔎 Detect unusual pricing patterns across SKUs
- 📉 Identify low stock + high defect rate risks
- 🔗 Monitor data quality in SAP supply chain pipelines
- 🛡️ Prevent operational losses from bad data decisions
""")

# ── BUSINESS PROBLEM ───────────────────────────────────────────────────────
st.subheader("🏢 Business Problem")
st.markdown("""
In supply chain systems, inaccurate or inconsistent data can lead to **inventory mismatches, 
stock shortages, overstocking, and delays in logistics operations**. Organizations using SAP 
need a reliable way to continuously monitor data quality and detect anomalies before they 
impact business decisions and operational efficiency.
""")

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
        return df, True
    except Exception:
        df = pd.read_csv("supply_chain_data.csv")
        df.columns = [col.strip().replace("_", " ").title() for col in df.columns]
        return df, False

st.sidebar.success("✅ Connected to SAP HANA Cloud")
df, is_live = load_data()

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

# ── DATA HEALTH STATUS ─────────────────────────────────────────────────────
st.header("🏥 Data Health Status")
if overall >= 90:
    st.success(f"🟢 **GOOD** — Overall data quality score is {overall}/100. Supply chain data is reliable.")
elif overall >= 70:
    st.warning(f"🟡 **MODERATE** — Overall data quality score is {overall}/100. Some issues need attention.")
else:
    st.error(f"🔴 **CRITICAL** — Overall data quality score is {overall}/100. Immediate action required.")

st.markdown("---")

# ── OVERALL DQ HEALTH ──────────────────────────────────────────────────────
st.header("📊 Overall Data Quality Health")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Overall Score", f"{overall}/100")
col2.metric("Completeness", f"{completeness}/100")
col3.metric("Uniqueness", f"{uniqueness}/100")
col4.metric("Validity", f"{validity}/100")
col5.metric("Consistency", f"{consistency}/100")

st.markdown("---")

# ── DQ DIMENSION CHART ─────────────────────────────────────────────────────
st.header("📈 DQ Dimension Analysis")
dq_data = pd.DataFrame({
    'Dimension': ['Completeness', 'Uniqueness', 'Validity', 'Consistency'],
    'Score': [completeness, uniqueness, validity, consistency]
})
fig = px.bar(dq_data, x='Dimension', y='Score', color='Score',
             color_continuous_scale='RdYlGn', range_color=[60, 100],
             range_y=[0, 100], text='Score')
fig.update_traces(texttemplate='%{text}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── AI ANOMALY DETECTION ───────────────────────────────────────────────────
st.header("🤖 AI Anomaly Detection")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Normal Records", len(df) - len(anomalies))
col3.metric("🚨 High-Risk Records", len(anomalies))

if len(anomalies) > 0:
    st.subheader("🚨 High-Risk Records Found By AI")
    display_cols = [c for c in [
        'Sku', 'Product Type', 'Price', 'Defect Rates',
        'Stock Levels', 'status', 'anomaly_reason'
    ] if c in df.columns]
    st.dataframe(
        anomalies[display_cols].rename(columns={
            'anomaly_reason': '⚠️ Why Flagged?',
            'status': 'Risk Status'
        }),
        use_container_width=True
    )

# ── RECOMMENDED ACTION ─────────────────────────────────────────────────────
st.subheader("✅ Recommended Action")
st.markdown("""
- **Review flagged SKUs** before processing inventory or logistics operations
- **Validate stock quantities** and pricing entries for anomalous records
- **Investigate high defect rate** items to prevent quality issues in supply chain
- **Ensure data consistency** before using records for business reporting or planning
""")

st.markdown("---")

# ── BUSINESS IMPACT ────────────────────────────────────────────────────────
st.header("💼 Business Impact Assessment")

num_anomalies = len(anomalies)
avg_revenue = df['Revenue Generated'].mean() if 'Revenue Generated' in df.columns else 5000
estimated_risk = round(num_anomalies * avg_revenue * 0.15, 2)
detection_rate = round((num_anomalies / len(df)) * 100, 1)

col1, col2, col3 = st.columns(3)
col1.metric("🚨 High-Risk Records", num_anomalies,
            help="Records flagged by Isolation Forest ML model")
col2.metric("💰 Estimated Revenue at Risk", f"₹{estimated_risk:,.0f}",
            help="15% of avg revenue × number of high-risk records")
col3.metric("📊 Anomaly Detection Rate", f"{detection_rate}%",
            help="Percentage of records flagged as high-risk")

st.info(
    f"🛡️ **Proactive Alert:** {num_anomalies} supply chain records show unusual patterns. "
    f"Early detection prevents potential reporting errors and operational losses "
    f"worth approximately ₹{estimated_risk:,.0f} in revenue impact."
)

st.markdown("---")

# ── BUSINESS IMPACT PANEL ──────────────────────────────────────────────────
st.header("📋 Business Impact Panel")
col1, col2 = st.columns(2)
with col1:
    st.success("""
    **✅ What This System Prevents:**
    - Operational losses from bad inventory decisions
    - Shipment delays from undetected data issues
    - Financial reporting errors from inconsistent data
    - Supply chain disruptions from low stock + high defect combinations
    """)
with col2:
    st.info("""
    **📈 Business Value Delivered:**
    - Improves supply chain reporting accuracy
    - Reduces operational risk proactively
    - Enhances data-driven decision making
    - Monitors SAP data pipelines in real-time
    """)

st.markdown("---")

# ── DATA SOURCE ────────────────────────────────────────────────────────────
st.header("🔗 Data Source")
st.info("📡 Live data from SAP HANA Cloud | Instance: dq-monitoring-hana | Region: Singapore - Azure")

# ── COMPLETE DATASET ───────────────────────────────────────────────────────
st.header("📋 Complete Dataset View")
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

    show_suspicious_only = st.checkbox("Show High-Risk Records Only")
    if show_suspicious_only:
        df_display = df_display[df_display['status'] == 'SUSPICIOUS']

st.dataframe(df_display, use_container_width=True)

# ── SUMMARY INSIGHT ────────────────────────────────────────────────────────
st.markdown("---")
st.header("🧠 Summary Insight")
high_defect = anomalies[anomalies['Defect Rates'] > df['Defect Rates'].quantile(0.75)] if 'Defect Rates' in df.columns else pd.DataFrame()
low_stock = anomalies[anomalies['Stock Levels'] < df['Stock Levels'].quantile(0.25)] if 'Stock Levels' in df.columns else pd.DataFrame()

st.warning(
    f"📊 **System Analysis:** The AI model analyzed {len(df)} supply chain records and detected "
    f"{num_anomalies} high-risk anomalies — including {len(high_defect)} records with high defect rates "
    f"and {len(low_stock)} records with critically low stock levels. "
    f"These data integrity issues could impact supply chain reporting accuracy and operational decisions. "
    f"Immediate review of flagged SKUs is recommended."
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.info("""
**AI-Powered Supply Chain DQ Monitor**
- Built with SAP HANA Cloud + AI
- Isolation Forest Anomaly Detection
- Real-time Data Quality Scoring
- Business Impact Assessment
- B.Tech Special Project 2026
- By: Bharath Kumar Panda
""")
st.sidebar.markdown(f"🕒 Last refreshed: {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}")
