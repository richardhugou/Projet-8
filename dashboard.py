import streamlit as st

st.set_page_config(page_title="Prêt à Dépenser - Portail", layout="centered")

st.title("Prêt à Dépenser - Portail d'Accès")
st.markdown("---")

st.markdown("""
Bienvenue sur le portail de l'établissement financier **Prêt à Dépenser**.

Veuillez sélectionner l'espace correspondant à votre profil dans le **menu de navigation (à gauche)**.
""")

st.write("")
st.write("")

col1, col2 = st.columns(2)

with col1:
    st.info(
        "**Espace Conseiller Clientèle**\n\n"
        "- Simulez ou uploadez un dossier de crédit.\n"
        "- Obtenez la décision instantanée de l'IA.\n"
        "- Accédez à la fiche d'interprétabilité locale (Facteurs d'influence)."
    )

with col2:
    st.warning(
        "**Tour de Contrôle MLOps (Admin)**\n\n"
        "- Consultez les métriques de production.\n"
        "- Surveillez l'analyse de Data Drift.\n"
        "- Opérez la bascule à chaud du Modèle (ZDT Hot-Swap)."
    )
