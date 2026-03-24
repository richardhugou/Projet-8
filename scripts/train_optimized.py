import os
import gc
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import time
import mlflow
import mlflow.lightgbm
import re

# 1. Configuration et chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'internal_doc', 'Projet 6', 'Projet+Mise+en+prod+-+home-credit-default-risk', 'application_train.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

COST_FN = 10
COST_FP = 1

COLUMNS_TO_CLIP = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']

def run_training(df_full, top_n, output_path, experiment_name="Projet8_CreditScoring"):
    print(f"\n--- Entraînement du modèle avec Top {top_n} features ---")
    
    # Copie locale pour ne pas polluer les autres runs
    df = df_full.copy()
    
    # 4. Pré-traitement de base (Encodage et Séparation X/y)
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    
    # Encodage One-Hot des variables catégorielles
    cat_columns = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_columns, drop_first=True)
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    feature_names = X.columns.tolist()
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Imputation globale par la médiane
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    X_train_df = pd.DataFrame(X_train_imputed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_imputed, columns=feature_names)
    
    # Sélection Automatique du Top N via un premier run LightGBM
    lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=-1)
    lgb_selector.fit(X_train_df, y_train)
    
    importances = pd.Series(lgb_selector.feature_importances_, index=feature_names)
    top_features = importances.nlargest(top_n).index.tolist()
    
    X_train_top = X_train_df[top_features]
    X_val_top = X_val_df[top_features]
    
    # Calcul des statistiques (Moyenne et Écart-type) pour les features sélectionnées
    # Ces stats sont calculées sur les données imputées mais AVANT SMOTE
    feature_stats = {}
    for col in top_features:
        feature_stats[col] = {
            'mean': float(X_train_top[col].mean()),
            'std': float(X_train_top[col].std())
        }
    
    # Ré-entrainement de l'imputeur UNIQUEMENT sur les variables sélectionnées
    final_imputer = SimpleImputer(strategy='median')
    X_train_top_arr = final_imputer.fit_transform(X_train_top)
    X_val_top_arr = final_imputer.transform(X_val_top)
    
    # Équilibrage SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_top_arr, y_train)
    
    # Modèle Final avec MLFlow
    best_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"Top_{top_n}_Features"):
        mlflow.log_params(best_params)
        mlflow.log_param("top_n", top_n)
        
        final_model = lgb.LGBMClassifier(**best_params)
        start_time = time.time()
        final_model.fit(X_train_resampled, y_train_resampled)
        train_time = time.time() - start_time
        
        # Optimisation du Seuil
        y_proba = final_model.predict_proba(X_val_top_arr)[:, 1]
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        min_cost = float('inf')
        
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            # Gestion du cas où une seule classe est prédite (rare ici)
            cm = confusion_matrix(y_val, y_pred_t)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                cost = COST_FN * fn + COST_FP * fp
                if cost < min_cost:
                    min_cost = cost
                    best_threshold = t
                    
        y_pred_best = (y_proba >= best_threshold).astype(int)
        cm = confusion_matrix(y_val, y_pred_best)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        
        auc = roc_auc_score(y_val, y_proba)
        
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("min_business_cost", min_cost)
        mlflow.log_metric("roc_auc", auc)
        mlflow.lightgbm.log_model(final_model, "model")
        
        print(f"Top {top_n} - ROC AUC : {auc:.4f}, Seuil : {best_threshold:.3f}")
        
        # Sauvegarde de l'artefact complet
        export_dict = {
            'model': final_model,
            'imputer': final_imputer,
            'features': top_features,
            'feature_stats': feature_stats,
            'metrics': {
                'best_threshold': best_threshold,
                'min_business_cost': min_cost,
                'roc_auc': auc,
                'training_time': train_time
            }
        }
        joblib.dump(export_dict, output_path)
        print(f"✅ Modèle sauvegardé : {output_path}")

def main():
    print("--- Démarrage de la Phase de Réduction de Features ---")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Fichier de données introuvable : {DATA_PATH}")
        
    print(f"Chargement de {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 3. Traitement des Outliers (Kaggle Home Credit)
    print("Traitement des outliers...")
    # Cas spécifique DAYS_EMPLOYED
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Clipping pour les variables financières (basé sur le 99ème percentile pour limiter l'impact des extrêmes)
    for col in COLUMNS_TO_CLIP:
        if col in df.columns:
            upper_limit = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=upper_limit)
            print(f"  - Clipping effectué pour {col} (max: {upper_limit:.2f})")
    
    # Création des variables 'Domain Knowledge'
    print("Création des Domain Knowledge Features...")
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365
    df['YEARS_BIRTH'] = df['DAYS_BIRTH'] / -365
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Lancement des deux entraînements
    run_training(df, 15, os.path.join(MODEL_DIR, 'scoring_model_15.joblib'))
    run_training(df, 50, os.path.join(MODEL_DIR, 'scoring_model_50.joblib'))
    
    # Mise à jour du lien symbolique ou du fichier par défaut si nécessaire
    # Par défaut, on garde scoring_model.joblib pointant vers le 50 pour la rétro-compatibilité
    # ou on écrase le 50 sur scoring_model.joblib
    import shutil
    shutil.copy(os.path.join(MODEL_DIR, 'scoring_model_50.joblib'), os.path.join(MODEL_DIR, 'scoring_model.joblib'))
    
    print("\n✅ Tous les modèles ont été générés avec succès.")

if __name__ == "__main__":
    main()
