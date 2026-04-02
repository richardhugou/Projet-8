import pytest
from fastapi.testclient import TestClient

# On importe l'app et le dictionnaire d'artefacts pour vérifier le chargement
from api.main import app, ml_artifacts


def test_model_is_loaded():
    """Vérifier que le modèle est bien monté en RAM au démarrage."""
    with TestClient(app) as _:
        # Si le chargement Lifespan a fonctionné, le dictionnaire ne doit pas être vide
        assert len(ml_artifacts) > 0
        assert "model" in ml_artifacts
        assert "features" in ml_artifacts
        print(f"\n[OK] Modèle chargé avec {len(ml_artifacts['features'])} features.")


def test_essential_bricks_availability():
    """Vérifier que les composants essentiels (bricks) sont importables et prêts."""
    try:
        import pandas as pd  # noqa: F401
        import joblib  # noqa: F401
        import sklearn  # noqa: F401
        import lightgbm  # noqa: F401

        # Si on arrive ici, l'environnement est prêt pour la suite de l'exercice
        assert True
    except ImportError as e:
        pytest.fail(f"Une brique essentielle est manquante : {e}")
