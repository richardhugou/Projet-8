#!/bin/bash
set -m

# Lancement de FastAPI en tâche de fond (Port Local 8000)
# L'API ne sera pas accessible directement depuis l'extérieur sur Hugging Face,
# mais Streamlit (qui tourne dans le même conteneur) pourra la joindre via http://localhost:8000
echo "Démarrage de l'API FastAPI..."
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Attente pour s'assurer que l'API est démarrée avant le Front
sleep 3

# Lancement de Streamlit en processus principal (Port Public 7860)
# Le port 7860 est le seul port ouvert par défaut par un conteneur Hugging Face Spaces.
echo "Démarrage du Dashboard Streamlit..."
uv run streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0 --server.headless true
