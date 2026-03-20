# Roadmap du Projet : Scoring de Crédit avec FastAPI, uv et GitHub Actions

## Phase 0 : Création d'un modèle simplifié (Pipeline Préliminaire)
*Objectif : Réduire le nombre de features du modèle du Projet 6 pour faciliter l'intégration et l'explicabilité de l'API.*
- [ ] Analyser les notebooks du Projet 6 pour extraire les features les plus importantes (Top Features).
- [ ] Créer un script d'entraînement simplifié (`scripts/train_optimized.py`) ne gardant que ces features.
- [ ] Entraîner et évaluer ce modèle "allégé".
- [ ] Exporter ce nouveau modèle (`model/scoring_model.joblib`) pour la production.

## Phase 1 : Préparation du dépôt et du modèle
- [ ] Exporter le modèle final : `CreditScoring_LGBM_SMOTE_Best` depuis MLflow (format joblib ou pickle).
- [x] Initialiser le dépôt Git : Fait (`git init`, `main`, `develop`, `feature/ci-cd`).
- [x] Créer l'arborescence du projet : Dossiers `api/`, `tests/`, `monitoring/` et `model/`.
- [ ] Placer le modèle exporté dans le dossier `model/`.
- [x] Créer le fichier `.gitignore` : Initialisé (à compléter avec `internal_doc/`).
- [x] Premier commit de base : Fait.

## Phase 2 : Initialisation de l'environnement et API (FastAPI)
- [x] Initialiser le projet avec `uv` : Fait (`uv init`).
- [ ] Ajouter les dépendances : `uv add fastapi uvicorn pandas lightgbm pydantic`.
- [ ] Créer le fichier `api/main.py`.
- [ ] Définir le schéma d'entrée (Pydantic).
- [ ] Implémenter le chargement du modèle (singleton au démarrage).
- [ ] Créer la route de prédiction : `POST /predict`.
- [ ] Implémenter le système de logs : `logs_production.csv`.
- [ ] Tester l'API en local : `uv run uvicorn api.main:app --reload`.

## Phase 3 : Tests automatisés et Conteneurisation
- [x] Ajouter les dépendances de développement : Fait (`uv add --dev pytest`).
- [ ] Ajouter `httpx` pour les tests API.
- [ ] Créer le fichier `tests/test_api.py`.
- [ ] Écrire les tests unitaires (cas valides, données manquantes, types incorrects).
- [x] Lancer les tests en local : Fait (`uv run pytest tests/test_env.py`).
- [ ] Créer le `Dockerfile` optimisé pour `uv`.
- [ ] Construire l'image Docker : `docker build -t scoring-api .`.

## Phase 4 : CI/CD (Pipeline d'automatisation)
- [x] Créer l'arborescence `.github/workflows/` : Fait.
- [x] Créer `ci.yml` (anciennement `main.yml`) : Placeholder fonctionnel avec `setup-uv`.
- [ ] Configurer le déclencheur sur `push` vers `main`.
- [ ] Configurer le Job de Test : `uv sync` + `uv run pytest`.
- [ ] Configurer le Job de Build Docker (conditionné aux tests).
- [ ] Configurer le Job de Déploiement.

## Phase 5 : Monitoring et Data Drift
- [ ] Ajouter les dépendances : `uv add evidently streamlit`.
- [ ] Simuler des données d'utilisation pour `logs_production.csv`.
- [ ] Captures d'écran pour livrables.
- [ ] Créer `monitoring/drift_analysis.py`.
- [ ] Implémenter la comparaison avec Evidently (référence vs production).
- [ ] Générer le rapport Evidently (HTML).
- [ ] Créer le Dashboard Streamlit : `monitoring/app.py`.

## Phase 6 : Profiling et Optimisation
- [ ] Utiliser `cProfile` sur la route de prédiction.
- [ ] Analyser et identifier les goulots d'étranglement.
- [ ] Documenter l'optimisation proposée.

## Phase 7 : Finalisation et Soumission
- [ ] Rédiger le `README.md` final.
- [ ] Documenter les commandes (`Docker`, `uv`, `pytest`, `monitoring`).
- [ ] Vérifier la clarté de l'historique Git.
- [x] Pousser le code : En cours (feature branches).
- [ ] Vérifier le pipeline CI/CD final.
- [ ] Remplir la fiche d'auto-évaluation.
