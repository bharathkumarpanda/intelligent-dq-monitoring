# 🔍 AI-Powered Supply Chain Data Quality Monitor

### Detects anomalies in inventory, pricing and defect data to prevent operational losses

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen)](https://bharath-dq-monitor.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)](https://streamlit.io)
[![SAP HANA](https://img.shields.io/badge/SAP%20HANA-Cloud-orange)](https://www.sap.com)

---

## 🚀 Live Demo
👉 **[https://bharath-dq-monitor.streamlit.app](https://bharath-dq-monitor.streamlit.app)**

---

## 📌 Business Problem
In supply chain systems, inaccurate or inconsistent data can lead to inventory mismatches, stock shortages, overstocking, and delays in logistics operations. Organizations using SAP need a reliable way to continuously monitor data quality and detect anomalies before they impact business decisions and operational efficiency.

---

## 🧠 What This System Does
- **Real-time Data Quality Scoring** — Monitors Completeness, Uniqueness, Validity and Consistency
- **AI Anomaly Detection** — Uses Isolation Forest ML model to detect high-risk supply chain records
- **Explainable AI** — Shows exactly WHY each record was flagged (not just that it was flagged)
- **Business Impact Assessment** — Quantifies revenue at risk from detected anomalies
- **Recommended Actions** — Guides users on what to do after anomalies are detected

---

## 📦 Use Case: Supply Chain Data Monitoring
- 🔎 Detect unusual pricing patterns across SKUs
- 📉 Identify low stock + high defect rate risks
- 🔗 Monitor data quality in SAP supply chain pipelines
- 🛡️ Prevent operational losses from bad data decisions

---

## 🛠️ Tech Stack
| Technology | Purpose |
|-----------|---------|
| Python | Core programming language |
| Streamlit | Web application framework |
| SAP HANA Cloud | Primary data source (cloud database) |
| Isolation Forest (scikit-learn) | ML anomaly detection |
| Plotly | Interactive data visualizations |
| Pandas | Data processing and analysis |

---

## ✨ Key Features
- 🏥 **Data Health Status** — 🟢 Good / 🟡 Moderate / 🔴 Critical badge
- 📊 **DQ Dimension Analysis** — Completeness, Uniqueness, Validity, Consistency scores
- 🤖 **AI Anomaly Detection** — Isolation Forest with 5% contamination rate
- ⚠️ **Why Flagged?** — Explainable reasoning for each suspicious record
- 💼 **Business Impact Panel** — Revenue at risk + detection rate
- 🔽 **Filter Controls** — Filter by product type or show high-risk records only
- 🕒 **Last Refreshed Timestamp** — Always know when data was last updated

---

## 📁 Project Structure
```
intelligent-dq-monitoring/
│
├── dashboard.py          # Main Streamlit app
├── ai_model.py           # Isolation Forest anomaly detection
├── dq_engine.py          # Data quality scoring logic
├── data_loader.py        # Data loading utilities
├── sap_connector.py      # SAP HANA Cloud connection
├── supply_chain_data.csv # Sample supply chain dataset
└── requirements.txt      # Python dependencies
```

---

## ⚙️ How to Run Locally
```bash
# Clone the repository
git clone https://github.com/bharathkumarpanda/intelligent-dq-monitoring.git

# Navigate to project folder
cd intelligent-dq-monitoring

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run dashboard.py
```

---

## 📊 Sample Results
- **100 supply chain records** analyzed
- **5 high-risk anomalies** detected
- **₹4,332 estimated revenue** at risk
- **5% anomaly detection rate**

---

## 👨‍💻 About
**Bharath Kumar Panda**
 B.Tech student at ICFAI Tech, Hyderabad.
Self-taught SAP HANA Cloud developer passionate about 
AI and supply chain data quality.

---

## 📬 Connect
- 🔗 [GitHub](https://github.com/bharathkumarpanda)
- 💼 [LinkedIn](https://www.linkedin.com/in/bharath-kumar-panda-ab2053345)
