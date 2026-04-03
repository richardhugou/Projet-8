import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="P7 - Monitoring Score Crédit",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style Custom (Premium Dark Mode feel)
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4a4a4a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "production_inference.jsonl")
REPORT_HTML = os.path.join(BASE_DIR, "monitoring", "reports", "drift_report.html")
REPORT_JSON = os.path.join(BASE_DIR, "monitoring", "reports", "drift_report.json")


def load_data():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()

    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            row = {
                "timestamp": entry["timestamp"],
                "probability": entry["outputs"]["probability_default"],
                "status": entry["outputs"]["status"],
                "latency": entry["latency_ms"],
            }
            data.append(row)
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# --- SIDEBAR ---
with st.sidebar:
    st.title("Contrôle MLOps")
    st.info("Ce dashboard surveille l'API de Scoring en temps réel.")

    if st.button("Rafraîchir les données"):
        st.rerun()

# --- MAIN CONTENT ---
st.title("Prêt à Dépenser")
st.markdown("---")

df = load_data()

if df.empty:
    st.warning("Aucun log de production détecté. Lancez d'abord la simulation.")
else:
    # 1. KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Requêtes", len(df))
    with col2:
        avg_lat = round(df["latency"].mean(), 2)
        st.metric(
            "Latence Moyenne",
            f"{avg_lat}ms",
            delta=f"{round(avg_lat - 20, 1)}ms vs target",
            delta_color="inverse",
        )
    with col3:
        # Calcul robuste : filtrage + comptage des lignes
        count_acc = df[df["status"] == "ACCORDÉ"].shape[0]
        acc_rate = round((count_acc / len(df)) * 100, 1)
        st.metric("Taux d'Accord", f"{acc_rate}%")
    with col4:
        # Calcul robuste
        count_ref = df[df["status"] == "REFUSÉ"].shape[0]
        ref_rate = round((count_ref / len(df)) * 100, 1)
        st.metric("Taux de Refus", f"{ref_rate}%")

    # 2. Graphiques de Performance
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Latence d'Inférence (ms)")
        fig_lat = px.line(
            df,
            x="timestamp",
            y="latency",
            template="plotly_dark",
            color_discrete_sequence=["#00f2ff"],
        )
        fig_lat.add_hline(
            y=50, line_dash="dash", line_color="red", annotation_text="SLA 50ms"
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_right:
        st.subheader("Distribution des Scores")
        fig_hist = px.histogram(
            df,
            x="probability",
            color="status",
            barmode="overlay",
            template="plotly_dark",
            color_discrete_map={"ACCORDÉ": "#00cc96", "REFUSÉ": "#ef553b"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # 3. Drift Analysis (Evidently Integration)
    st.markdown("---")
    st.header("Analyse de Dérive (Data Drift)")

    tab1, tab2 = st.tabs(["Rapport Interactif", "Données Brutes (JSON)"])

    with tab1:
        if os.path.exists(REPORT_HTML):
            with open(REPORT_HTML, "r", encoding="utf-8") as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=800, scrolling=True)
        else:
            st.info(
                "Le rapport HTML n'a pas encore été généré par `drift_analysis.py`."
            )

    with tab2:
        if os.path.exists(REPORT_JSON):
            with open(REPORT_JSON, "r") as f:
                drift_data = json.load(f)
            st.json(drift_data)
        else:
            st.info("Le rapport JSON est introuvable.")

# Footer
st.markdown("---")
st.caption(
    f"Dernière mise à jour : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Projet 8 - MLOps Industrialisation"
)
