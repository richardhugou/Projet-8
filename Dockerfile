# Utilisation de l'image UV optimisée pour la phase de construction
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# Configuration UV pour accélérer le build et compiler le bytecode
ENV UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des dépendances (mise en cache des couches Docker)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Ajout du code source et du modèle
COPY api/ /app/api/
COPY model/ /app/model/
COPY README.md /app/

# Installation du projet final sans les dépendances de dev
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# --- IMAGE FINALE ---
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copie de l'environnement virtuel et de l'application depuis le builder
COPY --from=builder /app /app

# Ajout de l'environnement virtuel au PATH
ENV PATH="/app/.venv/bin:$PATH"

# Exposition du port FastAPI
EXPOSE 8000

# Commande de démarrage avec Uvicorn
# Utilisation de 0.0.0.0 pour l'accès externe au conteneur
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
