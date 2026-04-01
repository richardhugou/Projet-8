import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Espace Conseiller", layout="wide")

API_URL = "http://localhost:8000"  # Port par défaut de notre FastAPI Dockerisé

st.title("Espace Conseiller Clientèle")
st.markdown("---")


def display_client_fiche(response_json):
    status = response_json.get("status", "INCONNU")
    proba = response_json.get("probability_default", 0)

    st.header("Résultat de l'Analyse d'Octroi")

    col_gauge, col_shap = st.columns([1, 1])

    with col_gauge:
        # Jauge Premium Plotly
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Risque de Défaut (%)", "font": {"size": 24}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {
                        "color": "rgba(0,0,0,0)"
                    },  # Barre remplacée par le curseur (threshold)
                    "steps": [
                        {
                            "range": [
                                0,
                                response_json.get("threshold_used", 0.5) * 100,
                            ],
                            "color": "rgba(43, 201, 144, 0.4)",
                        },  # Vert doux (Accordé)
                        {
                            "range": [
                                response_json.get("threshold_used", 0.5) * 100,
                                100,
                            ],
                            "color": "rgba(255, 102, 102, 0.4)",
                        },  # Rouge doux (Refusé)
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 6},
                        "thickness": 0.75,
                        "value": proba * 100,
                    },
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # Affichage texte de la conclusion
        if status == "ACCORDÉ":
            st.success("CRÉDIT ACCORDÉ - Risque faible")
        else:
            st.error("CRÉDIT REFUSÉ - Risque de non-remboursement jugé trop élevé")

    with col_shap:
        st.subheader("Interprétabilité de l'IA (Top 3 Facteurs)")
        top_features = response_json.get("top_features_impact", [])
        
        # Dictionnaire de traduction (Technique -> Métier)
        TRADUCTIONS = {
            "EXT_SOURCE_1": "Score Externe 1",
            "EXT_SOURCE_2": "Score Externe 2",
            "EXT_SOURCE_3": "Score Externe 3",
            "DAYS_BIRTH": "Âge du Client",
            "AMT_ANNUITY": "Annuité Demandée",
            "AMT_CREDIT": "Montant du Crédit",
            "AMT_INCOME_TOTAL": "Revenus du Client",
            "DAYS_EMPLOYED": "Ancienneté Emploi",
            "CODE_GENDER": "Genre",
            "NAME_EDUCATION_TYPE": "Niveau d'Éducation",
            "NAME_FAMILY_STATUS": "Statut Familial"
        }
        
        if top_features:
            # Transformation automatique en bar chart pour les non-Data Scientists
            df_shap = pd.DataFrame(top_features)
            
            # Application de la traduction
            df_shap['Nom Critique'] = df_shap['feature'].apply(lambda x: TRADUCTIONS.get(x, x.replace("_", " ").title()))
            
            # Label métier : Positif (SHAP > 0) veut dire que ça augmente le risque de défaut !
            df_shap['Impact'] = df_shap['shap_value'].apply(lambda x: 'Augmente le Risque' if x > 0 else 'Baisse le Risque')
            
            fig_shap = px.bar(
                df_shap, 
                x="shap_value", 
                y="Nom Critique", 
                orientation='h',
                color="Impact",
                color_discrete_map={'Augmente le Risque': '#ff6666', 'Baisse le Risque': '#2bc990'},
            )
            
            # Design : On inverse Y pour mettre la plus forte influence tout en haut
            fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)

            st.info(
                "*Note pour le conseiller* : Ces 3 variables ont eu le plus grand "
                "poids mathématique pour justifier la décision d'octroi de l'IA envers ce client spécifique."
            )
        else:
            st.info(
                "Aucune interprétation locale n'a été renvoyée par le modèle actuel."
            )


# Moteur de Saisie Modulaire (Concept "Dual Workflow")
mode = st.radio(
    "Comment souhaitez-vous saisir le dossier ?",
    ["Saisie Manuelle Simplifiée", "Upload d'un Fichier Client (CSV)"],
    horizontal=True,
)

if mode == "Saisie Manuelle Simplifiée":
    st.markdown(
        "**(Démonstration) L'API imputera automatiquement par la médiane les variables non-saisies.**"
    )
    with st.form("manual_form"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            amt_annuity = st.number_input("Montant de l'Annuité demandée ($)", value=25000, step=1000)
            age_annees = st.slider("Âge du Client (en années)", min_value=18, max_value=90, value=40)
            ext_source_1 = st.slider("Score Crédit Externe 1", min_value=0.0, max_value=1.0, value=0.5)
        with col_m2:
            ext_source_2 = st.slider("Score Crédit Externe 2", min_value=0.0, max_value=1.0, value=0.5)
            ext_source_3 = st.slider("Score Crédit Externe 3", min_value=0.0, max_value=1.0, value=0.5)
            
        submit = st.form_submit_button("Lancer l'Analyse Risque", type="primary")
        
        if submit:
            # Conversion métier transparente pour l'API (Années -> Jours négatifs)
            days_birth_converted = int(-age_annees * 365.25)
            
            payload = {
                "AMT_ANNUITY": amt_annuity,
                "DAYS_BIRTH": days_birth_converted,
                "EXT_SOURCE_1": ext_source_1,
                "EXT_SOURCE_2": ext_source_2,
                "EXT_SOURCE_3": ext_source_3
            }
            with st.spinner("Vérification en cours par l'IA..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload)
                    if resp.status_code == 200:
                        display_client_fiche(resp.json())
                    else:
                        st.error(f"Erreur de l'API ({resp.status_code}) : {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error(
                        f"Impossible de contacter l'API sur {API_URL}. FastAPI est-il allumé ?"
                    )

else:
    st.info(
        "Uploadez le fichier CSV d'un client (toutes les colonnes requises doivent être présentes)."
    )
    uploaded_file = st.file_uploader("Dossier Client (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(1))  # On montre la première ligne

        if st.button("Calculer le score du premier client", type="primary"):
            # On envoie la première ligne convertie en dictionnaire
            client_dict = df.iloc[0].to_dict()
            # Les NaN pandas peuvent faire crasher JSON, on remplace par null ou on laisse l'API géré (Pydantic accepte Null)
            # Simplification: suppression des vrais NaN
            client_dict_clean = {k: v for k, v in client_dict.items() if pd.notna(v)}

            with st.spinner("Analyse du client de la ligne 0..."):
                try:
                    resp = requests.post(f"{API_URL}/predict", json=client_dict_clean)
                    if resp.status_code == 200:
                        display_client_fiche(resp.json())
                    else:
                        st.error(f"Erreur API ({resp.status_code}) : {resp.text}")
                except Exception as e:
                    st.error(f"Erreur de connexion : {e}")
