# Mission : Prêt à Dépenser - Mise en Production du Modèle de Scoring

## Contexte du Projet
En tant que Data Scientist chez "Prêt à Dépenser", l'objectif est de mettre en production un modèle de scoring (précédemment développé avec MLflow) pour le département "Crédit Express". L'API doit être robuste, déployable via Docker et monitorée en temps réel.

## Objectifs Principaux
1.  **Simplification du Modèle (Phase 0)** : Ré-entraîner le modèle sur un sous-ensemble des variables les plus pertinentes pour faciliter l'intégration de l'API.
2.  **API de Scoring** : Développer une API (FastAPI) pour l'inférence en quasi temps réel.
2.  **Conteneurisation** : Créer un Dockerfile pour un déploiement fluide.
3.  **CI/CD** : Mettre en œuvre un pipeline automatisé (GitHub Actions) incluant tests et build Docker.
4.  **Monitoring & Data Drift** : Suivre les performances en production (Evidently, Streamlit) et détecter la dérive des données.
5.  **Optimisation** : Identifier et corriger les goulots d'étranglement de l'API (cProfile).

## Livrables Attendus par Chloé Dubois (Lead Data Scientist)
- [ ] **Historique Git** : Liste de commits clairs sur GitHub.
- [ ] **API Fonctionnelle** : Route `POST /predict` prenant des données client et retournant un score.
- [ ] **Tests Unitaires** : Tests automatisés robustes (FastAPI / Pytest).
- [ ] **Dockerfile** : Pour la conteneurisation de l'API.
- [ ] **Analyse du Data Drift** : Dashboard Streamlit ou rapport Evidently comparant entraînement et production.
- [ ] **Livrables de Stockage** : Screenshots de la solution de logs (`logs_production.csv`).
- [ ] **Pipeline CI/CD** : Fichier YAML automatisant tests, build et déploiement.
- [ ] **README** : Documentation d'installation et d'utilisation.

## Contraintes Techniques
- **Framework API** : FastAPI (choix par défaut) ou Gradio.
- **Gestionnaire de dépendances** : `uv` (exigence projet spécifique).
- **Chargement du modèle** :singleton au démarrage (ne pas recharger à chaque requête).
- **Dérive de données** : Utilisation d'Evidently AI ou NannyML avec une base de référence.
- **Sécurité** : Ne jamais commiter de données sensibles ou credentials.
