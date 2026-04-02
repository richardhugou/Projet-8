import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_handles_null_with_imputation():
    """
    Vérifie que l'API accepte les 'null' (None) grâce à Optional[float]
    et les traite via son imputation Médiane interne sans planter.
    """
    # Payload avec des valeurs nulles (NaN) pour tester la résilience
    payload = {
        "EXT_SOURCE_1": None,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": None,
        "DAYS_BIRTH": -15000,
        "AMT_ANNUITY": None,  # Test critique sur un montant
        "AMT_CREDIT": 500000,
        "AMT_INCOME_TOTAL": 200000,
        "DAYS_ID_PUBLISH": -3000,
        "DAYS_LAST_PHONE_CHANGE": -100,
        "DAYS_EMPLOYED": -2000,
    }

    # L'API ne doit pas renvoyer 422 (Unprocessable Entity)
    response = client.post("/predict", json=payload)

    if response.status_code == 503:
        pytest.skip("Modèle non chargé (Service Unavailable), test ignoré.")

    assert response.status_code == 200, f"L'API a rejeté les null : {response.text}"

    data = response.json()
    assert "probability_default" in data
    assert "status" in data

    print(
        f"Test API Null Handling : SUCCÈS (Status: {data['status']}, Proba: {data['probability_default']:.3f})"
    )


if __name__ == "__main__":
    # Exécution manuelle pour validation rapide
    try:
        test_api_handles_null_with_imputation()
    except Exception as e:
        print(f"Erreur lors du test : {e}")
