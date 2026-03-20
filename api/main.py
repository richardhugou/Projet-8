import os
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging

from api.schemas import ClientData

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_api")

# Pattern Singleton pour les artefacts ML :
# On utilise un dictionnaire global pour stocker le modèle et ses dépendances (imputeur, features...).
# Ces éléments sont chargés UNE SEULE FOIS lors du démarrage de l'API (startup) et réutilisés
# à chaque requête, évitant ainsi des chargements disque/CPU coûteux.
ml_artifacts = {}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'scoring_model.joblib')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage (Startup) - Chargement unique des artefacts en mémoire vive (RAM)
    logger.info("Tentative de chargement du modèle depuis {}".format(MODEL_PATH))
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Fichier modèle introuvable ! L'API démarrera mais les prédictions renverront 503.")
    else:
        # Singleton : Le joblib.load n'est exécuté qu'ici, au lancement du processus
        artefact = joblib.load(MODEL_PATH)
        ml_artifacts['model'] = artefact['model']
        ml_artifacts['imputer'] = artefact['imputer']
        ml_artifacts['features'] = artefact['features']
        ml_artifacts['threshold'] = artefact['metrics']['best_threshold']
        logger.info("Modèle et métriques chargés avec succès dans la RAM (Singleton prêt).")
        
    yield
    
    # Extinction (Shutdown)
    logger.info("Libération de la mémoire vive (RAM)...")
    ml_artifacts.clear()

app = FastAPI(
    title="Prêt à Dépenser - API de Scoring",
    description="API MLOps pour l'octroi de crédit",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API de Scoring opérationnelle."}

@app.post("/predict")
def predict_credit(client: ClientData):
    if not ml_artifacts:
        # Modèle manquant (Testé unitairement)
        raise HTTPException(status_code=503, detail="Modèle non disponible. Veuillez contacter l'administrateur.")
        
    try:
        client_dict = client.model_dump()
        df_client = pd.DataFrame([client_dict], columns=ml_artifacts['features'])
        
        X_imputed = ml_artifacts['imputer'].transform(df_client)
        
        proba_default = ml_artifacts['model'].predict_proba(X_imputed)[0, 1]
        threshold = ml_artifacts['threshold']
        prediction = 1 if proba_default >= threshold else 0
        
        status = "REFUSÉ" if prediction == 1 else "ACCORDÉ"
        
        return {
            "probability_default": float(proba_default),
            "threshold_used": float(threshold),
            "prediction": int(prediction),
            "status": status
        }
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur lors de la prédiction.")
