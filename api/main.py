import os
import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
import logging
import json
import time
import shutil
from datetime import datetime, timezone

from api.schemas import ClientData

# Configuration du logging standard (Console/Info)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_api")

# Configuration du logging PRODUCTION (JSON Lines pour Monitoring)
# Ce fichier contiendra l'historique complet pour l'analyse de Data Drift
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Stratégie de Persistance Cloud (Hugging Face Storage Bucket)
CLOUD_STORAGE_DIR = "/data"
IS_CLOUD = os.path.exists(CLOUD_STORAGE_DIR)

if IS_CLOUD:
    logger.info(
        "Dossier persistant Cloud détecté (/data). Activation du mode Production."
    )
    LOG_DIR = os.path.join(CLOUD_STORAGE_DIR, "logs")
else:
    logger.info("Utilisation des répertoires de développement locaux.")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "production_inference.jsonl")


def log_prediction(inputs: dict, outputs: dict, latency: float, status_code: int = 200):
    """Enregistre une ligne de log structurée en JSON (Fichier + Console)."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
        "outputs": outputs,
        "latency_ms": round(latency * 1000, 2),
        "status_code": status_code,
        "model_version": ml_artifacts.get("version", "unknown_version"),
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

MODEL_FILENAME = os.getenv("SCORING_MODEL_FILENAME", "optuna_scoring_model.joblib")

if IS_CLOUD:
    MODEL_DIR = os.path.join(CLOUD_STORAGE_DIR, "model")
else:
    MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Démarrage (Startup) - Chargement unique des artefacts en mémoire vive (RAM)
    logger.info("Tentative de chargement du modèle depuis {}".format(MODEL_PATH))

    # S'assurer que le dossier logs existe
    os.makedirs(LOG_DIR, exist_ok=True)

    # Amorçage (Seeding) du Bucket Persistant (Cloud)
    # Si on est sur Hugging Face et que le bucket est vide, on y copie le modèle initial fournit par Github
    if IS_CLOUD and not os.path.exists(MODEL_PATH):
        original_model_path = os.path.join(BASE_DIR, "model", MODEL_FILENAME)
        if os.path.exists(original_model_path):
            logger.info(
                f"Amorçage du Storage Bucket : Copie du modèle initial vers {MODEL_PATH}"
            )
            shutil.copy2(original_model_path, MODEL_PATH)

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
        ml_artifacts["version"] = datetime.now(timezone.utc).isoformat()

        # Initialisation de SHAP (Mise en cache pour performances)
        try:
            ml_artifacts["explainer"] = shap.TreeExplainer(ml_artifacts["model"])
        except Exception as e:
            logger.warning(f"Impossible d'initialiser SHAP: {e}")
            ml_artifacts["explainer"] = None

        logger.info(
            f"Modèle '{MODEL_FILENAME}' chargé avec {len(ml_artifacts['features'])} features (Top {len(ml_artifacts['features'])}). Seuil: {ml_artifacts['threshold']:.3f}. Version: {ml_artifacts['version']}"
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
    return {
        "status": "ok",
        "message": "API de Scoring opérationnelle.",
        "model_version": ml_artifacts.get("version", "non chargé"),
    }


@app.post("/admin/update_model")
async def update_model(file: UploadFile = File(...)):
    """Route Admin pour mettre à jour le modèle à chaud (Hot-Swap) avec Zéro Downtime."""
    if not file.filename.endswith(".joblib"):
        raise HTTPException(
            status_code=400, detail="Le fichier doit être au format .joblib"
        )

    temp_model_path = MODEL_PATH + ".tmp"

    # 1. Sauvegarde temporaire (ne touche pas le modèle actif)
    try:
        content = await file.read()
        with open(temp_model_path, "wb") as f:
            f.write(content)
        logger.info(
            f"HOT-SWAP STEP 1 : Fichier {file.filename} sauvegardé en temporaire sur {temp_model_path}"
        )
    except Exception as e:
        logger.error(f"Erreur d'écriture du modèle : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur I/O: {str(e)}")

    # 2. Validation complète avant remplacement du modèle actif
    try:
        nouveau_artefact = joblib.load(temp_model_path)

        # Vérification d'intégrité minimale de l'artefact
        required_keys = {"model", "imputer", "features", "metrics"}
        if not required_keys.issubset(nouveau_artefact.keys()):
            missing = sorted(required_keys.difference(set(nouveau_artefact.keys())))
            raise ValueError(f"L'artefact est incomplet. Clés manquantes : {missing}")

        if "best_threshold" not in nouveau_artefact["metrics"]:
            raise ValueError(
                "L'artefact ne contient pas la clé metrics['best_threshold']."
            )

        if not isinstance(nouveau_artefact["features"], list) or not nouveau_artefact[
            "features"
        ]:
            raise ValueError("La liste des features est invalide ou vide.")

        # Remplacement atomique du modèle actif sur disque
        os.replace(temp_model_path, MODEL_PATH)

        # Remplacement atomique dans le dictionnaire Singleton
        ml_artifacts["model"] = nouveau_artefact["model"]
        ml_artifacts["imputer"] = nouveau_artefact["imputer"]
        ml_artifacts["features"] = nouveau_artefact["features"]
        ml_artifacts["threshold"] = nouveau_artefact["metrics"]["best_threshold"]
        ml_artifacts["version"] = datetime.now(timezone.utc).isoformat()

        try:
            ml_artifacts["explainer"] = shap.TreeExplainer(ml_artifacts["model"])
        except Exception as e:
            logger.warning(f"Impossible d'initialiser SHAP post-hot-swap: {e}")
            ml_artifacts["explainer"] = None

        logger.warning(
            f"HOT-SWAP RÉUSSI Nouvelle version activée : {ml_artifacts['version']}"
        )

        return {
            "status": "success",
            "message": "Bascule à chaud réussie.",
            "version": ml_artifacts["version"],
            "n_features": len(ml_artifacts["features"]),
        }

    except Exception as e:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)
        logger.error(f"HOT-SWAP ÉCHEC : {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Le modèle est corrompu ou illisible : {str(e)}"
        )


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

        # Interprétabilité Locale (SHAP)
        top_features = []
        if ml_artifacts.get("explainer"):
            try:
                shap_vals = ml_artifacts["explainer"].shap_values(X_imputed)
                # Gestion des différentes structures de retour selon l'algorithme (LightGBM vs autres)
                if isinstance(shap_vals, list):
                    client_shap = (
                        shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                    )
                elif len(shap_vals.shape) == 3:
                    client_shap = shap_vals[0, :, 1]
                else:
                    client_shap = shap_vals[0]

                # Extraction des Top 3 Impacts
                feat_names = ml_artifacts["features"]
                shap_dict = {
                    feat_names[i]: float(client_shap[i]) for i in range(len(feat_names))
                }
                sorted_shap = sorted(
                    shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
                )
                top_features = [
                    {"feature": k, "shap_value": v} for k, v in sorted_shap[:3]
                ]
            except Exception as e:
                logger.error(f"Erreur de calcul SHAP: {e}")

        # Préparation réponse
        response_data = {
            "probability_default": float(proba_default),
            "threshold_used": float(threshold),
            "prediction": int(prediction),
            "status": status,
            "top_features_impact": top_features,
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
