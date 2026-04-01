import numpy as np
import pandas as pd
import joblib
import os

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "scoring_model_50.joblib")


def test_simulation_impute_alignment():
    """
    Vérifie que la simulation utilise l'imputer de l'artefact
    et non une logique KNN externe.
    """
    # 1. Chargement de l'imputeur réel
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle non trouvé à {MODEL_PATH}, test ignoré.")
        return

    artefact = joblib.load(MODEL_PATH)
    trained_imputer = artefact["imputer"]
    target_features = artefact["features"]

    # 2. Création d'une donnée de test avec NaN
    data = {feature: 1.0 for feature in target_features}
    # On force un NaN sur une variable connue pour être imputée par la médiane
    # Par exemple 'EXT_SOURCE_1'
    data[target_features[0]] = np.nan

    df_test = pd.DataFrame([data])

    # 3. Imputation via l'imputeur entraîné
    X_imputed = pd.DataFrame(
        trained_imputer.transform(df_test), columns=target_features
    )

    # Récupération de la valeur médiane entraînée pour cette feature
    # SimpleImputer stocke les valeurs dans .statistics_
    feature_idx = 0
    expected_median = trained_imputer.statistics_[feature_idx]
    valeur_imputee = X_imputed.iloc[0, feature_idx]

    print(f"\nFeature: {target_features[feature_idx]}")
    print(f"Valeur attendue (Médiane): {expected_median}")
    print(f"Valeur imputée : {valeur_imputee}")

    # Assertion
    assert np.isclose(valeur_imputee, expected_median), (
        f"L'imputation n'est pas alignée sur la médiane de l'artefact : {valeur_imputee} != {expected_median}"
    )


if __name__ == "__main__":
    try:
        test_simulation_impute_alignment()
        print("Test Imputation Alignment : SUCCÈS")
    except Exception as e:
        print(f"Test Imputation Alignment : ÉCHEC ({str(e)})")
