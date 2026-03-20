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
        # Les données factices (le schéma Pydantic s'occupe des valeurs par défaut à 0.0)
        payload = {}
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 503
        assert "Modèle non disponible" in response.json()["detail"]

def test_predict_withModel_returns_200():
    """Teste la prédiction normale si le modèle est présent."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Modèle non trouvé à {MODEL_PATH}, test ignoré.")
        
    with TestClient(app) as client:
        # Payload vide (tout à 0) pour tester juste que ça passe l'imputeur et le ML
        payload = {}
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        json_resp = response.json()
        assert "probability_default" in json_resp
        assert "prediction" in json_resp
        assert "status" in json_resp
        assert json_resp["status"] in ["ACCORDÉ", "REFUSÉ"]
