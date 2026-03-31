---
title: Credit Scoring API (Projet 8)
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# Prêt à Dépenser - Scoring de Crédit MLOps

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Ce projet industriel (Projet 8) vise à déployer un modèle de scoring de crédit robuste, sécurisé et monitoré. L'objectif est de prédire la probabilité de défaut d'un client tout en minimisant le coût financier métier (optimisation du Recall).

---

## Fonctionnalités Clés

- **API Haute Performance** : Développée avec **FastAPI**, utilisant un chargement **Singleton** (Lifespan) pour une latence minimale.
- **Modèle Optimisé** : Basé sur **LightGBM** avec une sélection du Top 50 des variables les plus discriminantes.
- **Sécurité Bancaire** : Validation stricte des flux de données via **Pydantic**.
- **Monitoring Intégré** : Journalisation structurée en `JSON Lines` pour la détection du **Data Drift** via **Evidently AI**.
- **Industrialisation** : Conteneurisation **Docker** (multi-stage) et pipeline **CI/CD** (GitHub Actions).

---

## Structure du Projet

```text
.
├── api/                # Cœur de l'application (Vues & Schémas)
├── model/              # Artefacts ML (Modèle, Imputeur, Stats)
├── monitoring/         # Dashboard Streamlit & Analyse de Drift
├── scripts/            # Outils d'entraînement et de simulation
├── tests/              # Suite de tests (Unitaires & Readiness)
├── logs/               # Traces d'inférence (JSONL)
├── Dockerfile          # Image de production
└── pyproject.toml      # Gestion des dépendances (uv)
```

---

## Installation & Lancement QuickStart

Ce projet utilise **[`uv`](https://github.com/astral-sh/uv)** pour une gestion ultra-rapide des dépendances.

### 1. Initialisation
```bash
# Installation des dépendances
uv sync
```

### 2. Démarrer l'API de Scoring
```bash
# Lancement local (Port 8000)
uv run uvicorn api.main:app --reload
```
Accédez à la documentation interactive : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Lancer le Dashboard de Monitoring (Phase 6)
```bash
uv run streamlit run monitoring/app.py
```

---

## Architecture Technique (Rationnel)

### Le Chargement Singleton
Pour éviter de saturer la RAM et de ralentir l'API, le modèle est chargé **une seule fois** au démarrage. Chaque requête pioche directement dans les artefacts résidents en mémoire, garantissant une réponse en moins de 50ms.

### Le Seuil de Décision Métier
Nous avons choisi de favoriser le **Recall** par rapport à la Précision. Un Faux Négatif coûtant 10x plus cher qu'un Faux Positif, l'API rejette le crédit dès que la probabilité de défaut dépasse **~10%** (seuil optimal calculé lors de l'entraînement).

### Validation & Schémas
Toutes les données entrantes sont vérifiées par `api/schemas.py`. Si une donnée est hors plage (ex: revenu négatif) ou d'un type erroné, l'API renvoie immédiatement une erreur **422**, protégeant ainsi le moteur de calcul métier.

---

## Docker
Pour construire et lancer l'image de production :
```bash
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api
```

---

## Tests Unitaires
Pour exécuter la suite complète de 9 tests (API + Readiness) :
```bash
uv run python3 -m pytest
```

---
*Projet réalisé dans le cadre du parcours Data Scientist - Projet n°8.*
