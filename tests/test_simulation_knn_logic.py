import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

def test_knn_imputation_logic():
    """
    Vérifie que le KNNImputer (K=2) remplace bien le NaN par la moyenne des voisins.
    C'est ce mécanisme qui est utilisé dans generate_production_data.py.
    """
    # 1. Création d'un mini-dataset (3 lignes, 2 colonnes)
    # Ligne 0 et 1 sont proches. Ligne 2 a un NaN.
    data = [
        [10.0, 20.0],
        [11.0, 21.0],
        [np.nan, 20.5] # Le NaN devrait être la moyenne de 10.0 et 11.0 (soit 10.5)
    ]
    df = pd.DataFrame(data, columns=['A', 'B'])
    
    # 2. Application du KNN (k=2)
    imputer = KNNImputer(n_neighbors=2)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=['A', 'B'])
    
    # 3. Assertions
    valeur_imputee = df_imputed.loc[2, 'A']
    expected = 10.5
    
    print(f"\nValeur attendue : {expected}")
    print(f"Valeur imputée par KNN : {valeur_imputee}")
    
    assert np.isclose(valeur_imputee, expected), f"L'imputation KNN a échoué : {valeur_imputee} != {expected}"
    assert not df_imputed.isnull().values.any(), "Il reste des valeurs nulles après imputation."

if __name__ == "__main__":
    try:
        test_knn_imputation_logic()
        print("✅ Test KNN Imputation : SUCCÈS")
    except Exception as e:
        print(f"❌ Test KNN Imputation : ÉCHEC ({str(e)})")
