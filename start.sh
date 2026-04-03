#!/bin/bash
set -euo pipefail

cleanup() {
	if [[ -n "${PID_API:-}" ]] && kill -0 "$PID_API" 2>/dev/null; then
		kill "$PID_API" 2>/dev/null || true
	fi
	if [[ -n "${PID_DASHBOARD:-}" ]] && kill -0 "$PID_DASHBOARD" 2>/dev/null; then
		kill "$PID_DASHBOARD" 2>/dev/null || true
	fi
}

trap cleanup EXIT INT TERM

# Lancement de FastAPI en tâche de fond (Port Local 8000)
# L'API ne sera pas accessible directement depuis l'extérieur sur Hugging Face,
# mais Streamlit (qui tourne dans le même conteneur) pourra la joindre via http://localhost:8000
echo "Démarrage de l'API FastAPI..."
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
PID_API=$!

# Attente pour s'assurer que l'API est démarrée avant le Front
sleep 3

if ! kill -0 "$PID_API" 2>/dev/null; then
	echo "Échec: l'API FastAPI s'est arrêtée au démarrage."
	exit 1
fi

# Lancement de Streamlit en processus principal (Port Public 7860)
# Le port 7860 est le seul port ouvert par défaut par un conteneur Hugging Face Spaces.
echo "Démarrage du Dashboard Streamlit..."
python -m streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0 --server.headless true &
PID_DASHBOARD=$!

# Si un des deux processus meurt, on coupe l'autre pour éviter un état zombie.
wait -n "$PID_API" "$PID_DASHBOARD"
EXIT_CODE=$?
echo "Un service s'est arrêté (code ${EXIT_CODE}). Arrêt contrôlé du conteneur."
exit "$EXIT_CODE"
