import os
import pytest
import json
from unittest.mock import patch
from fastapi.testclient import TestClient

from api.main import app, MODEL_PATH

# Chemin temporaire pour cacher le modèle lors du test de résilience
MODEL_HIDDEN_PATH = MODEL_PATH + ".hidden"

def test_read_root():
    # Avec TestClient, le asyncontextmanager s'exécute, donc l'API démarre.
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "message": "API de Scoring opérationnelle."}

def test_predict_missing_model_returns_503():
    """Teste si l'API renvoie 503 quand le modèle n'est pas trouvé (via Mocking)."""
    # On patche le chemin du modèle vers un dossier inexistant
    with patch("api.main.MODEL_PATH", "/tmp/non_existent_folder/missing_model.joblib"):
        with TestClient(app) as client:
            # Données valides au sens Pydantic mais modèle absent
            payload = {
                "AMT_INCOME_TOTAL": 100000,
                "AMT_CREDIT": 500000,
                "AMT_ANNUITY": 25000,
                "DAYS_BIRTH": -15000,
                "EXT_SOURCE_1": 0.5,
                "EXT_SOURCE_2": 0.5,
                "EXT_SOURCE_3": 0.5
            }
            
            response = client.post("/predict", json=payload)
            assert response.status_code == 503
            assert "Modèle non disponible" in response.json()["detail"]

def test_predict_withModel_returns_200():
    """Teste la prédiction normale si le modèle est présent."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Modèle non trouvé à {MODEL_PATH}, test ignoré.")
        
    with TestClient(app) as client:
        # Payload valide
        payload = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 25000,
            "DAYS_BIRTH": -15000,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.5
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        json_resp = response.json()
        assert "probability_default" in json_resp
        assert "prediction" in json_resp
        assert "status" in json_resp
        assert json_resp["status"] in ["ACCORDÉ", "REFUSÉ"]

def test_predict_missing_mandatory_fields_returns_422():
    """Vérifie que l'API rejette les requêtes sans les champs obligatoires."""
    with TestClient(app) as client:
        # On oublie les champs obligatoires
        payload = {"AMT_ANNUITY": 1000, "DAYS_BIRTH": -10000}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

def test_predict_invalid_types_returns_422():
    """Vérifie que l'API rejette les types de données incorrects."""
    with TestClient(app) as client:
        payload = {
            "AMT_INCOME_TOTAL": "beaucoup d'argent", # String au lieu de float
            "AMT_CREDIT": 50000,
            "AMT_ANNUITY": 2000,
            "DAYS_BIRTH": -15000,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.5
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

def test_predict_out_of_range_values_returns_422():
    """Vérifie que les contraintes métier (ex: ge=0) sont respectées."""
    with TestClient(app) as client:
        payload = {
            "AMT_INCOME_TOTAL": -100, # Négatif interdit
            "AMT_CREDIT": 50000,
            "AMT_ANNUITY": 2000,
            "DAYS_BIRTH": -15000,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.5
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        
        # Test DAYS_BIRTH positif (interdit par le schéma le=0)
        payload["AMT_INCOME_TOTAL"] = 50000
        payload["DAYS_BIRTH"] = 500 # Positif interdit
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

def test_predict_generates_log_file():
    """Vérifie qu'une ligne de log est bien écrite lors d'une prédiction."""
    from api.main import LOG_FILE
    
    # On s'assure d'un état propre : supprimer le log s'il existe
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    with TestClient(app) as client:
        payload = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 25000,
            "DAYS_BIRTH": -15000,
            "EXT_SOURCE_1": 0.5,
            "EXT_SOURCE_2": 0.5,
            "EXT_SOURCE_3": 0.5
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        # Vérifier l'existence et le contenu du fichier
        assert os.path.exists(LOG_FILE)
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            assert len(lines) >= 1
            last_log = json.loads(lines[-1])
            assert "timestamp" in last_log
            assert "latency_ms" in last_log
            assert last_log["inputs"]["AMT_INCOME_TOTAL"] == 100000
