import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import json
import time
from datetime import datetime

from api.schemas import ClientData

# Configuration du logging standard (Console/Info)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_api")

# Configuration du logging PRODUCTION (JSON Lines pour Monitoring)
# Ce fichier contiendra l'historique complet pour l'analyse de Data Drift
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "production_inference.jsonl")


def log_prediction(inputs: dict, outputs: dict, latency: float, status_code: int = 200):
    """Enregistre une ligne de log structurée en JSON (Fichier + Console)."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "latency_ms": round(latency * 1000, 2),
        "status_code": status_code,
    }
    
    # 1. Stockage physique (Exigence Projet 8 - Screenshots)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    # 2. Sortie Standard (Best Practice Docker / Observabilité) Cette brique permet dans le cadre d'un déploiement de voir les logs dans le terminal ou Docker Logs
    logger.info(f"PRODUCTION_LOG: {json.dumps(log_entry)}")


# Pattern Singleton pour les artefacts ML :
# On utilise un dictionnaire global pour stocker le modèle et ses dépendances (imputeur, features...).
# Ces éléments sont chargés UNE SEULE FOIS lors du démarrage de l'API (startup) et réutilisés
# à chaque requête, évitant ainsi des chargements disque/CPU coûteux.
ml_artifacts = {}

MODEL_FILENAME = os.getenv("SCORING_MODEL_FILENAME", "scoring_model.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "model", MODEL_FILENAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage (Startup) - Chargement unique des artefacts en mémoire vive (RAM)
    logger.info("Tentative de chargement du modèle depuis {}".format(MODEL_PATH))

    # S'assurer que le dossier logs existe
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Fichier modèle introuvable ! L'API démarrera mais les prédictions renverront 503."
        )
    else:
        # Singleton : Le joblib.load n'est exécuté qu'ici, au lancement du processus
        artefact = joblib.load(MODEL_PATH)
        ml_artifacts["model"] = artefact["model"]
        ml_artifacts["imputer"] = artefact["imputer"]
        ml_artifacts["features"] = artefact["features"]
        ml_artifacts["threshold"] = artefact["metrics"]["best_threshold"]
        logger.info(
            f"Modèle '{MODEL_FILENAME}' chargé avec {len(ml_artifacts['features'])} features (Top {len(ml_artifacts['features'])}). Seuil: {ml_artifacts['threshold']:.3f}."
        )

    yield

    # Extinction (Shutdown)
    logger.info("Libération de la mémoire vive (RAM)...")
    ml_artifacts.clear()


app = FastAPI(
    title="Prêt à Dépenser - API de Scoring",
    description="API MLOps pour l'octroi de crédit",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de Scoring opérationnelle."}


@app.post("/predict")
def predict_credit(client: ClientData):
    if not ml_artifacts:
        # Modèle manquant (Testé unitairement)
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Veuillez contacter l'administrateur.",
        )

    start_time = time.perf_counter()
    try:
        client_dict = client.model_dump()
        df_client = pd.DataFrame([client_dict], columns=ml_artifacts["features"])

        # Inférence
        X_imputed = ml_artifacts["imputer"].transform(df_client)
        proba_default = ml_artifacts["model"].predict_proba(X_imputed)[0, 1]

        threshold = ml_artifacts["threshold"]
        prediction = 1 if proba_default >= threshold else 0
        status = "REFUSÉ" if prediction == 1 else "ACCORDÉ"

        # Préparation réponse
        response_data = {
            "probability_default": float(proba_default),
            "threshold_used": float(threshold),
            "prediction": int(prediction),
            "status": status,
        }

        # Logging de production (Monitoring)
        latency = time.perf_counter() - start_time
        log_prediction(client_dict, response_data, latency)

        return response_data

    except Exception as e:
        latency = time.perf_counter() - start_time
        logger.error(f"Erreur prédiction: {str(e)}")
        log_prediction(client.model_dump(), {"error": str(e)}, latency, status_code=500)
        raise HTTPException(
            status_code=500, detail="Erreur interne du serveur lors de la prédiction."
        )
