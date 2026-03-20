import os
import shutil
import pytest
from fastapi.testclient import TestClient

from api.main import app, BASE_DIR, MODEL_PATH

# Chemin temporaire pour cacher le modèle lors du test de résilience
MODEL_HIDDEN_PATH = MODEL_PATH + ".hidden"

@pytest.fixture
def hide_model():
    """Déplace temporairement le modèle pour simuler son absence."""
    was_hidden = False
    if os.path.exists(MODEL_PATH):
        shutil.move(MODEL_PATH, MODEL_HIDDEN_PATH)
        was_hidden = True
        
    yield
    
    # Rétablir le modèle après le test
    if was_hidden and os.path.exists(MODEL_HIDDEN_PATH):
        shutil.move(MODEL_HIDDEN_PATH, MODEL_PATH)

def test_read_root():
    # Avec TestClient, le asyncontextmanager s'exécute, donc l'API démarre.
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "message": "API de Scoring opérationnelle."}

def test_predict_missing_model_returns_503(hide_model):
    """Teste si l'API renvoie 503 quand le modèle n'est pas chargé."""
    with TestClient(app) as client:
        # Données valides au sens Pydantic mais modèle absent
        payload = {
            "AMT_INCOME_TOTAL": 100000,
            "AMT_CREDIT": 500000,
            "AMT_ANNUITY": 25000,
            "DAYS_BIRTH": -15000
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
            "DAYS_BIRTH": -15000
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
        # On oublie AMT_INCOME_TOTAL et AMT_CREDIT
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
            "DAYS_BIRTH": -15000
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
            "DAYS_BIRTH": -15000
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        
        # Test DAYS_BIRTH positif (interdit par le schéma le=0)
        payload["AMT_INCOME_TOTAL"] = 50000
        payload["DAYS_BIRTH"] = 500 # Positif interdit
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
