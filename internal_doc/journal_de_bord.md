# Journal de Bord - Projet 8 (Score Crédit MLOps)

Ce journal retrace chronologiquement les décisions techniques, les difficultés rencontrées et les solutions adoptées pour transformer l'approche modélisation (Projet 6) en véritable architecture de production (Projet 8).

## Phase 0 : De l'Expérimentation à l'Industrialisation

### 1. Le problème du volume de données
Lors de la conception de l'API (Projet 8), nous avons fait face à un défi lié au contrat d'interface. Le modèle brut issu du Projet 6 s'appuyait sur plusieurs centaines de variables issues de jointures lourdes (entre `application_train`, `bureau`, `credit_card_balance`, etc.). 
- **Difficulté :** Reproduire et envoyer l'intégralité de ces jointures à chaque appel réseau de l'API était inenvisageable en production (problèmes de latence, complexité de maintenance, payload massif).

### 2. Le choix du "Modèle Allégé" (Light MLOps)
Afin de concilier la performance de prédiction et la performance logicielle, il a été décidé de **ré-entraîner un modèle cible optimisé (seuil métier) mais drastiquement allégé**.
- **Source d'autorité** : On se focalise sur les variables ayant le plus haut pouvoir de séparation. L'analyse révèle que le cœur prédictif se situe dans le fichier natif `application_train.csv` (notamment `EXT_SOURCE_1`, `2`, `3`) couplé à de puissantes variables "Domain Knowledge" définies dans le Projet 6 (`CREDIT_TERM`, `ANNUITY_INCOME_PERCENT`, etc.).
- **Nouvelle approche :** Un script d'entraînement `train_optimized.py` (format MLOps) a été créé. Il charge __uniquement__ `application_train.csv`, en calcule les quelques variables métiers, sélectionne empiriquement le Top 50 des caractéristiques, équilibre les classes via `SMOTE`, et entraîne un `LGBMClassifier`.

### 3. La fonction d'Optimisation Financière (Coût Métier)
Dans le domaine bancaire, un Faux Négatif (client qui fait défaut mais à qui l'on a prêté) coûte infiniment plus cher (10x plus) qu'un Faux Positif (client sain à qui l'on a refusé un crédit sans raison).
Le script d'entraînement :
- Balaye 100 seuils de classification entre 0.00 et 1.00.
- Calcule la matrice de confusion et l'équation associée : `Coût Total = (10 * FN) + (1 * FP)`.
- Isole **le seuil de décision probabiliste exact qui minimise cette fuite financière**.
*(Lors du dernier run, le seuil optimal retenu fut `0.091`).*

### 4. Transparence, Traçabilité et Conservation Complète
Le grand défi du MLOps étant qu'un modèle "en prod" ou dont la RAM est vidée est une boîte noire, nous avons sécurisé deux niveaux de tracking :

1. **L'artefact enrichi (`model/scoring_model.joblib`) :** Au lieu de ne cracher qu'un modèle impénétrable, le script exporte un "dictionnaire/passeport". Outre l'algorithme LightGBM, il fige et stocke en dur : l'imputeur, le seuil optimisé, la liste immuable des 50 features dans leur ordre d'apparition, le temps d'entraînement et toutes les métriques (AUC, Recall, Precision). Un notebook test (`00_verify_exported_model.ipynb`) certifie que l'on peut rouvrir cet objet à tout moment pour la documentation.
2. **Le serveur expérimental MLflow :** Nous avons fait appel aux bonnes pratiques du Projet 6. Le script est englobé dans un bloc local MLflow (`Projet8_CreditScoring`). La run tagge le modèle, consigne la grille d'hyperparamètres et enregistre définitivement le Coût métier et l'AUC. Le tableau de bord est accessible via `uv run mlflow ui`.

### Prochaine Étape (Phase 1)
Notre "moteur" (artifact + MLflow + seuil) étant maintenant figé dans le marbre, la suite du projet consistera à exposer ce résultat via FastAPI et d'en containeriser son écoute via Docker.

## Phase 1 : Développement & Robustesse de l'API

### 1. Release v0.1.0 (Lab Lock)
Afin de marquer la fin de la période d'expérimentation et de figer le modèle, un tag Git `v0.1.0` a été créé. Ce tag constitue la "vérité terrain" du modèle allégé avant toute modification logicielle de l'API.

### 2. Conception de l'API "Résiliente" (Singleton)
L'API a été reconstruite sur la branche `feature/api`. Une attention particulière a été portée à la résilience logicielle :
- **FastAPI Lifespan** : Le modèle est chargé une seule fois au démarrage.
- **Gestion des pannes** : Si le fichier `scoring_model.joblib` est manquant ou corrompu au démarrage, l'API ne plante pas bêtement. Elle démarre en "mode dégradé" et renvoie une erreur explicite `HTTP 503 (Service Unavailable)` lors des tentatives de prédiction. Cela permet aux administrateurs de diagnostiquer un problème de montage de volume Docker sans que le serveur ne soit "down".

### 3. Tests Automatisés (Pytest)
Pour garantir la qualité du code, une suite de tests unitaires (`tests/test_api.py`) a été mise en place.
- **Incident de parcours** : Lors du premier lancement des tests, une erreur `ModuleNotFoundError` s'est produite. Cela est dû à la structure des dossiers Python (le module `api` n'était pas dans le `PYTHONPATH`).
- **Résolution** : L'utilisation de `python3 -m pytest` a résolu le problème en forçant l'ajout du répertoire courant au chemin de recherche des modules.
- **Couverture de Robustesse (Vigilance)** : La suite de tests a été étendue à 7 tests couvrant :
    - La route racine.
    - La résilience 503 (modèle absent) : **Refactorisé avec `unittest.mock.patch`** pour éviter de déplacer des fichiers sur le disque, garantissant une exécution sans effets de bord.
    - La validation Pydantic (422) : champs obligatoires manquants, types incorrects (texte), et valeurs aberrantes (revenu négatif ou âge positif).
    - L'écriture des logs JSON de production.
    - Un cas passant nominal (200 OK).

*(État actuel : 7 tests sur 7 réussis).*

### 4. Qualité du Code (Linter Ruff)
Pour satisfaire à l'exigence de qualité avant l'industrialisation, l'outil **Ruff** (v0.15.7) a été intégré aux dépendances de développement.
- **Résultats** : L'analyse initiale a révélé 6 incohérences mineures (imports inutilisés et f-strings superflus).
- **Action** : Le code a été automatiquement corrigé via `ruff check --fix`.
- **Statut** : Le projet est désormais conforme aux standards PEP8 et aux bonnes pratiques de sécurité.

## Phase 2 : Logging de Production et Monitoring

Pour répondre aux exigences de la mission "Data Scientist" de Prêt à Dépenser, l'API a été équipée d'un système de traçabilité complet.
- **Format** : JSON Lines (`.jsonl`) pour une ingestion facile par des outils d'analyse.
- **Données capturées** :
    - `timestamp` : Date et heure précise (ISO).
    - `inputs` : L'intégralité des 50 features client envoyées.
    - `outputs` : Probabilité de défaut, seuil utilisé et verdict (Accordé/Refusé).
    - `latency_ms` : Temps d'inférence mesuré précisément en millisecondes.
    - `status_code` : Statut HTTP (200, 503, 500).
- **Emplacement** : `logs/production_inference.jsonl` (configuré dans le `.gitignore` pour ne pas saturer le dépôt).

### Sécurité et Performance
- Le chargement du modèle reste **unique** (Singleton) au démarrage.
- Le calcul de latence utilise `time.perf_counter()` pour une précision maximale sans impact sur le temps de réponse.

## Phase 4 : Dockerisation de l'API (Industrialisation)

L'application a été containerisée pour garantir la portabilité et la reproductibilité totale de l'environnement :
- **Stratégie** : Build multi-stage pour minimiser la taille de l'image finale et optimiser le cache Docker.
- **Dockerfile** : 
    - **Étape 1 (Builder)** : Utilisation de `ghcr.io/astral-sh/uv` (image ultra-performante) pour installer les dépendances et compiler le bytecode Python.
    - **Étape 2 (Runtime)** : Image finale `python:3.12-slim-bookworm` ne contenant que le strict nécessaire au fonctionnement de l'API.
- **Optimisation** : Un fichier `.dockerignore` strict a été mis en place pour exclure les données brutes (`data/`), les notebooks de recherche et les documents internes du conteneur de production.
- **Paramétrage** : L'API écoute sur le port `8000` et est accessible via l'hôte `0.0.0.0` à l'intérieur du conteneur.

### 1. Validation E2E (Smoke Test)
Un script de test automatisé `scripts/test_docker_smoke.sh` a certifié l'intégralité de la chaîne.
- **Difficulté rencontrée** : L'image `python:3.12-slim` ne contient pas la bibliothèque **OpenMP** par défaut. Comme **LightGBM** (moteur du domaine métier) utilise cette bibliothèque pour paralléliser les calculs, l'API renvoyait une erreur `OSError: libgomp.so.1: cannot open shared object file`.
- **Résolution** : Installation explicite de `libgomp1` (implémentation GNU d'OpenMP) dans le Dockerfile. Sans cet outil, le moteur de calcul métier ne pourrait pas charger ses bibliothèques C++ natives.
- **Résultat du test** : Succès total du build (v4) et du démarrage du serveur. Inférence isolée réussie : `{"probability_default": 0.057, "status": "ACCORDÉ"}`.
- **Verdict** : Le domaine métier est parfaitement préservé dans l'environnement containerisé.

## Guide de Validation Technique

Pour que vous puissiez valider vous-même l'intégrité de mon travail sur la Phase 1, voici les deux méthodes de test :

### 1. Validation via Swagger (Interface Interactive)
Lancez l'API localement avec cette commande :
`uv run uvicorn api.main:app --reload`

Une fois le serveur démarré, ouvrez votre navigateur à l'adresse suivante :
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

Vous pourrez :
- Dérouler la section **POST /predict**.
- Cliquer sur **Try it out**.
- Cliquer sur **Execute** (avec le JSON par défaut tout à 0.0).
- Vérifier que l'API renvoie bien une `probability_default` et un `status` (ACCORDÉ/REFUSÉ) basé sur le seuil de **0.091**.

### 2. Validation Automatisée (Pytest)
Pour vérifier que la "mécanique" interne est solide (notamment la gestion d'absence du modèle), lancez :
`uv run python3 -m pytest tests/test_api.py`

*Note : La Phase 4 (Docker) est strictement en attente de votre validation de ces étapes.*
