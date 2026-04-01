import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import requests

st.set_page_config(page_title="Tour de Contrôle MLOps", layout="wide")

API_URL = "http://localhost:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "production_inference.jsonl")
REPORT_HTML = os.path.join(BASE_DIR, "monitoring", "reports", "drift_report.html")
REPORT_JSON = os.path.join(BASE_DIR, "monitoring", "reports", "drift_report.json")

st.title("Tour de Contrôle MLOps")
st.success("Accès Administrateur Autorisé.")
st.markdown("---")


def load_logs():
    if not os.path.exists(LOG_FILE):
        return pd.DataFrame()
    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                data.append(
                    {
                        "timestamp": pd.to_datetime(entry.get("timestamp")),
                        "latency": entry.get("latency_ms", 0),
                        "status": entry.get("outputs", {}).get("status", "INCONNU"),
                        "probability": entry.get("outputs", {}).get(
                            "probability_default", 0
                        ),
                        "model_version": entry.get("model_version", "unknown"),
                    }
                )
            except Exception:
                pass
    return pd.DataFrame(data)


df = load_logs()

# ================================
# SECTION 1: HOT SWAP ZERO DOWNTIME
# ================================
st.header("Model Registry & Hot-Swap (ZDT)")
st.info(
    "Poussez un modèle `.joblib` entraîné. Il sera formaté, stocké physiquement et injecté atomiquement en Mémoire RAM, remplaçant l'ancien modèle sans couper le serveur !"
)

try:
    api_health = requests.get(f"{API_URL}/")
    if api_health.status_code == 200:
        current_version = api_health.json().get("model_version", "Inconnue")
        st.metric("Version de Modèle Active en Production", current_version)
except Exception as e:
    st.error(f"Impossible de joindre l'API sur {API_URL}. Erreur : {e}")

uploaded_model = st.file_uploader(
    "Nouveau modèle à déployer (.joblib)", type=["joblib"]
)
if uploaded_model is not None:
    if st.button("Pousser en Production Mondiale", type="primary"):
        with st.spinner("Téléversement et Rechargement de la RAM en cours..."):
            files = {
                "file": (
                    uploaded_model.name,
                    uploaded_model.getvalue(),
                    "application/octet-stream",
                )
            }
            try:
                res = requests.post(f"{API_URL}/admin/update_model", files=files)
                if res.status_code == 200:
                    st.success(
                        f"Opération Réussie ! L'API tourne désormais sur la version : {res.json()['version']}"
                    )
                    st.balloons()
                else:
                    st.error(f"Échec ({res.status_code}) : {res.text}")
            except Exception as e:
                st.error(f"Erreur technique : {e}")

st.markdown("---")

# ================================
# SECTION 2: KPIS & MONITORING
# ================================
st.header("Statistiques d'Inférence")

if df.empty:
    st.warning(
        "Aucune prédiction enregistrée dans les logs. Simulez des clients pour voir les statistiques."
    )
else:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Inférences", len(df))
    with col2:
        avg_lat = round(df["latency"].mean(), 2)
        st.metric("Latence Moyenne", f"{avg_lat}ms")
    with col3:
        # Sécurité division par zéro
        acc_rate = (
            round((df[df["status"] == "ACCORDÉ"].shape[0] / len(df)) * 100, 1)
            if len(df) > 0
            else 0
        )
        st.metric("Taux d'Accord", f"{acc_rate}%")
    with col4:
        st.metric("Dernière Inférence", df["timestamp"].max().strftime("%H:%M:%S"))

    st.write(" ")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("Décisions par Version du Modèle")
        fig_status = px.histogram(
            df,
            x="model_version",
            color="status",
            barmode="group",
            color_discrete_map={"ACCORDÉ": "#2bc990", "REFUSÉ": "#ff6666"},
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with col_g2:
        st.subheader("Latence par Version")
        fig_lat = px.box(df, x="model_version", y="latency", color="model_version")
        st.plotly_chart(fig_lat, use_container_width=True)

# ================================
# SECTION 3: EVIDENTLY DRIFT (RENDU NATIF STREAMLIT)
# ================================
st.markdown("---")
st.header("Analyse de Data Drift (Native)")
st.write(
    "Voici l'état de santé des données en production comparé aux données d'entraînement, "
    "généré à partir de l'analyse evidently mais rendu nativement sans aucune rupture de design."
)

if os.path.exists(REPORT_JSON):
    with open(REPORT_JSON, "r") as f:
        drift_data = json.load(f)

    metrics = drift_data.get("metrics", [])

    # Extraction globale
    drifted_cols = 0
    total_cols = 0
    drift_share = 0.0

    # Extraction détaillée par variable
    drift_details = []

    for m in metrics:
        if m.get("metric_name", "").startswith("DriftedColumnsCount"):
            val = m.get("value", {})
            drifted_cols = int(val.get("count", 0))
            drift_share = float(val.get("share", 0.0))
        elif m.get("metric_name", "").startswith("ValueDrift"):
            col_name = m.get("config", {}).get("column", "Unknown")
            p_val = m.get("value")
            threshold = m.get("config", {}).get("threshold", 0.05)

            total_cols += 1
            if isinstance(p_val, float):
                is_drifted = p_val < threshold
                if is_drifted:
                    drift_details.append(
                        {
                            "Variable Impactée": col_name,
                            "Risque de Dérive (p-value)": f"{p_val:.5f}",
                            "Statut": "🚨 DÉRIVE DÉTECTÉE",
                        }
                    )

    # Affichage des KPIs Natifs
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.metric("Variables Surveillées", total_cols)
    with col_kpi2:
        st.metric(
            "Variables en Dérive",
            drifted_cols,
            f"+{drifted_cols} erreurs",
            delta_color="inverse",
        )
    with col_kpi3:
        st.metric("Taux global de dérive", f"{drift_share * 100:.1f}%")

    if drift_details:
        st.warning(
            f"Attention, {drifted_cols} variables commencent à avoir un profil statistique différent de l'entraînement."
        )
        df_drift = pd.DataFrame(drift_details)
        st.dataframe(df_drift, use_container_width=True, hide_index=True)
    else:
        st.success("Toutes les variables sont stables. Aucun Drift majeur détecté.")

    # Bouton de secours pour lire le rapport natif dans le navigateur sans iframe
    st.write("")
    if os.path.exists(REPORT_HTML):
        with open(REPORT_HTML, "r", encoding="utf-8") as f:
            html_data = f.read()
        st.download_button(
            label="Télécharger le graphique Evidently (Version HTML détaillée complète)",
            data=html_data,
            file_name="drift_report.html",
            mime="text/html",
        )
else:
    st.warning(
        "Le rapport de Drift n'a pas encore été généré. Démarrez l'API, effectuez des prédictions, puis exécutez `uv run python scripts/generate_production_data.py` suivi de `uv run python monitoring/drift_analysis.py`."
    )
