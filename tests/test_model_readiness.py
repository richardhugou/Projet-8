import os
import pytest
from fastapi.testclient import TestClient

# On importe l'app et le dictionnaire d'artefacts pour vérifier le chargement
from api.main import app, ml_artifacts

def test_model_is_loaded():
    """Vérifier que le modèle est bien monté en RAM au démarrage."""
    with TestClient(app) as client:
        # Si le chargement Lifespan a fonctionné, le dictionnaire ne doit pas être vide
        assert len(ml_artifacts) > 0
        assert "model" in ml_artifacts
        assert "features" in ml_artifacts
        print(f"\n[OK] Modèle chargé avec {len(ml_artifacts['features'])} features.")

def test_essential_bricks_availability():
    """Vérifier que les composants essentiels (bricks) sont importables et prêts."""
    try:
        import pandas as pd
        import joblib
        import sklearn
        import lightgbm
        # Si on arrive ici, l'environnement est prêt pour la suite de l'exercice
        assert True
    except ImportError as e:
        pytest.fail(f"Une brique essentielle est manquante : {e}")

# ==============================================================================
# PLACEHOLDERS POUR LA SUITE DE L'EXERCICE (Tests de prédiction métier)
# ==============================================================================

# TODO : Implémenter un test sur une dizaine de cas où le crédit doit être REFUSÉ
# On visera des dossiers avec un fort taux d'endettement ou des scores EXT_SOURCE bas.
# def test_predict_refusal_cases():
#     pass

# TODO : Implémenter un test sur une dizaine de cas où le crédit doit être ACCORDÉ
# On visera des profils stables avec revenus élevés et garanties solides.
# def test_predict_approval_cases():
#     pass

# TODO : Test de capacité de charge (Stress test léger)
# Vérifier la réponse de l'API sur un lot de 100 requêtes consécutives.
