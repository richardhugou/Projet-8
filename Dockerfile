# Utilisation de l'image UV optimisée
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Configuration UV
ENV UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Étape 1 : Copie des fichiers de dépendances et installation
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-dev

# Étape 2 : Ajout du code source et du modèle
COPY api/ /app/api/
COPY model/ /app/model/
COPY README.md /app/
COPY dashboard.py /app/
COPY pages/ /app/pages/
COPY monitoring/ /app/monitoring/
COPY start.sh /app/

# Installation finale du projet
RUN uv sync --frozen --no-dev

# --- IMAGE FINALE ---
FROM python:3.12-slim-bookworm

# Installation de libgomp1 (NÉCESSAIRE pour LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copie de l'environnement virtuel et de l'application depuis le builder
COPY --from=builder /app /app

# Ajout de l'environnement virtuel au PATH
ENV PATH="/app/.venv/bin:$PATH"

# Exposition du port public de Hugging Face (7860) et local (8000 interne)
EXPOSE 7860
EXPOSE 8000

# Commande de démarrage (Lance FastAPI et Streamlit)
CMD ["bash", "start.sh"]
