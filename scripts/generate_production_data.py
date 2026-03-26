import pandas as pd
import numpy as np
import joblib
import os
import re
from fastapi.testclient import TestClient
from api.main import app

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'internal_doc', 'Projet 6', 'Projet+Mise+en+prod+-+home-credit-default-risk', 'application_train.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data', 'application_test.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'scoring_model_50.joblib')
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'production_inference.jsonl')

def prepare_features(df_input):
    """Applique le même pré-traitement que Phase 0."""
    df = df_input.copy()
    
    # Outliers
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']:
        if col in df.columns:
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper)
        
    # Feature Engineering (Domain Knowledge)
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365
    df['YEARS_BIRTH'] = df['DAYS_BIRTH'] / -365
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    return df

def generate_simulation():
    print("--- Démarrage de la Simulation de Production (400+400+400) ---")
    
    client = TestClient(app)
    
    # Nettoyage des logs précédents
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        print(f"Ancien fichier {LOG_FILE} supprimé.")

    # 1. Chargement des Features cibles
    artefact = joblib.load(MODEL_PATH)
    target_features = artefact['features']
    
    # PHASE A : 400 clients de Référence (Train)
    print("Chargement de la référence (Train)...")
    df_train = pd.read_csv(TRAIN_DATA_PATH, nrows=1000) # On prend un pool
    df_train = prepare_features(df_train)
    
    # One-Hot Encoding partiel pour matcher les colonnes Top 50
    # (Simplified approach for simulation: we manually ensure columns match)
    X_train = pd.get_dummies(df_train)
    X_train = X_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    # On s'assure que toutes les features cibles sont présentes
    for col in target_features:
        if col not in X_train.columns:
            X_train[col] = 0
            
    print(f"Simulation Phase A : 400 appels de Référence (Train)...")
    for i in range(400):
        row = X_train[target_features].iloc[i].to_dict()
        # Conversion NaN -> None pour conformité JSON
        row = {k: (None if pd.isna(v) else float(v) if isinstance(v, (np.float64, np.int64)) else v) for k, v in row.items()}
        client.post("/predict", json=row)

    # PHASE B & C : Clients de Production (Test)
    print("Chargement de la production (Test)...")
    df_test = pd.read_csv(TEST_DATA_PATH, nrows=2000)
    df_test = prepare_features(df_test)
    X_test = pd.get_dummies(df_test)
    X_test = X_test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    for col in target_features:
        if col not in X_test.columns:
            X_test[col] = 0

    print(f"Simulation Phase B : 400 appels de Production (Test Nominale)...")
    for i in range(400):
        row = X_test[target_features].iloc[i].to_dict()
        row = {k: (None if pd.isna(v) else float(v) if isinstance(v, (np.float64, np.int64)) else v) for k, v in row.items()}
        client.post("/predict", json=row)

    print(f"Simulation Phase C : 400 appels avec Drift Salaire (*1.4)...")
    for i in range(400, 800):
        row = X_test[target_features].iloc[i].to_dict()
        if 'AMT_INCOME_TOTAL' in row and row['AMT_INCOME_TOTAL'] is not None:
            row['AMT_INCOME_TOTAL'] *= 1.4
            
        row = {k: (None if pd.isna(v) else float(v) if isinstance(v, (np.float64, np.int64)) else v) for k, v in row.items()}
        client.post("/predict", json=row)

    print(f"\n✅ Simulation terminée. {400+400+400} logs générés dans {LOG_FILE}.")

if __name__ == "__main__":
    generate_simulation()
