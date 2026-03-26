import pandas as pd
import numpy as np
import joblib
import os
import re
from fastapi.testclient import TestClient
from sklearn.impute import KNNImputer
from api.main import app

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'internal_doc', 'Projet 6', 'Projet+Mise+en+prod+-+home-credit-default-risk', 'application_train.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'application_test.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'scoring_model_50.joblib')
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'production_inference.jsonl')

def prepare_raw_features(df_input):
    """Prépare les colonnes de base et Domain Knowledge."""
    df = df_input.copy()
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Feature Engineering (Domain Knowledge)
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365
    df['YEARS_BIRTH'] = df['DAYS_BIRTH'] / -365
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df

def generate_simulation():
    print("--- Démarrage de la Simulation de Production (KNN Imputation) ---")
    
    client = TestClient(app)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Ancien fichier {LOG_FILE} supprimé.")

    # 1. Chargement des meta-données
    artefact = joblib.load(MODEL_PATH)
    target_features = artefact['features']
    
    # 2. Préparation des 3 segments (400 + 400 + 400)
    print("Préparation des 1200 lignes sources...")
    df_train = pd.read_csv(TRAIN_DATA_PATH, nrows=600)
    df_test = pd.read_csv(TEST_DATA_PATH, nrows=1200)
    
    # Segment A : Train (Base) - 400 premières lignes
    X_train_raw = prepare_raw_features(df_train.iloc[:400])
    # Segment B : Test (Normal) - 400 premières lignes
    X_test_raw = prepare_raw_features(df_test.iloc[:400])
    # Segment C : Test (Drift) - 400 lignes suivantes
    X_drift_raw = prepare_raw_features(df_test.iloc[400:800])
    X_drift_raw['AMT_INCOME_TOTAL'] *= 1.4
    
    # Fusion pour traitement global
    df_all = pd.concat([X_train_raw, X_test_raw, X_drift_raw], ignore_index=True)
    
    # Encodage One-Hot
    X_all = pd.get_dummies(df_all)
    X_all = X_all.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    # Alignement colonnes Top 50
    for col in target_features:
        if col not in X_all.columns:
            X_all[col] = np.nan # On laisse en NaN pour l'imputer KNN
            
    X_final = X_all[target_features].copy()
    
    # 3. Imputation KNN (n_neighbors=5)
    print("Application de KNNImputer (Calcul en cours)...")
    imputer = KNNImputer(n_neighbors=5)
    X_imputed_array = imputer.fit_transform(X_final)
    X_imputed = pd.DataFrame(X_imputed_array, columns=target_features)
    
    # 4. Envoi des requêtes à l'API via TestClient
    print(f"Simulation en cours (1200 appels)...")
    with TestClient(app) as client:
        for i in range(1200):
            row = X_imputed.iloc[i].to_dict()
            # On s'assure que tout est float pur (pas de numpy)
            row_json = {k: float(v) for k, v in row.items()}
            
            # Le TestClient déclenchera naturellement le logging dans api/main.py
            response = client.post("/predict", json=row_json)
            
            if response.status_code != 200:
                print(f"ALERTE : Requête {i} a échoué avec status {response.status_code}")
                print(response.json())
                break
            
            if (i+1) % 200 == 0:
                print(f"Progression : {i+1}/1200 appels terminés.")

    print(f"\n✅ Simulation terminée avec succès.")
    print(f"Fichier de logs généré : {LOG_FILE}")

if __name__ == "__main__":
    generate_simulation()
