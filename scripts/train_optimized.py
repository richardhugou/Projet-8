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

# 1. Configuration et chemins
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'internal_doc', 'Projet 6', 'Projet+Mise+en+prod+-+home-credit-default-risk', 'application_train.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, 'scoring_model.joblib')

# TOP_N : Le nombre de variables à conserver pour le modèle
TOP_N_FEATURES = 50
COST_FN = 10
COST_FP = 1

def main():
    print("--- Démarrage de la Phase 0 : Entraînement du modèle allégé ---")
    
    # 2. Chargement des données
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Fichier de données introuvable : {DATA_PATH}")
        
    print(f"Chargement de {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"Taille initiale : {df.shape}")
    
    # 3. Création des variables 'Domain Knowledge' (du Projet 6)
    print("Création des Domain Knowledge Features...")
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365
    df['YEARS_BIRTH'] = df['DAYS_BIRTH'] / -365
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # 4. Pré-traitement de base (Encodage et Séparation X/y)
    y = df['TARGET']
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    
    # Encodage One-Hot des variables catégorielles (strings)
    print("Encodage des variables catégorielles...")
    cat_columns = X.select_dtypes(include=['object']).columns.tolist()
    X = pd.get_dummies(X, columns=cat_columns, drop_first=True)
    
    import re
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    feature_names = X.columns.tolist()
    
    print("Séparation Train / Test...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Imputation globale par la médiane
    print("Imputation des valeurs manquantes (médiane)...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_val_imputed = imputer.transform(X_val)
    
    X_train_df = pd.DataFrame(X_train_imputed, columns=feature_names)
    X_val_df = pd.DataFrame(X_val_imputed, columns=feature_names)
    
    del X_train, X_val, df, X
    gc.collect()

    # 5. Sélection Automatique du "Top 50"
    print("Phase de sélection de variables : Entraînement LightGBM rapide...")
    lgb_selector = lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=-1)
    lgb_selector.fit(X_train_df, y_train)
    
    importances = pd.Series(lgb_selector.feature_importances_, index=feature_names)
    top_features = importances.nlargest(TOP_N_FEATURES).index.tolist()
    
    print(f"🚀 Top {TOP_N_FEATURES} Features conservées : {top_features[:5]}...")
    
    X_train_top = X_train_df[top_features]
    X_val_top = X_val_df[top_features]
    
    # Ré-entrainement de l'imputeur UNIQUEMENT sur les 50 variables (utile pour l'API en prod)
    final_imputer = SimpleImputer(strategy='median')
    X_train_top_arr = final_imputer.fit_transform(X_train_top)
    X_val_top_arr = final_imputer.transform(X_val_top)
    
    del X_train_df, X_val_df
    gc.collect()

    # 6. Équilibrage SMOTE
    print("Équilibrage des classes (SMOTE)...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_top_arr, y_train)

    # 7. Modèle Final avec MLFlow
    print("Entraînement du modèle final sur les 50 features...")
    best_params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 8,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # Configuration MLflow (dossier local par défaut)
    mlflow.set_experiment("Projet8_CreditScoring")
    
    with mlflow.start_run(run_name="Phase_0_Optimized_LightGBM"):
        mlflow.log_params(best_params)
        
        final_model = lgb.LGBMClassifier(**best_params)
        
        start_time = time.time()
        final_model.fit(X_train_resampled, y_train_resampled)
        train_time = time.time() - start_time
        print(f"Temps d'entraînement du modèle : {train_time:.2f} secondes")
        
        # 8. Évaluation et Optimisation du Seuil (Matrice de coût)
        print("Évaluation et Optimisation du Seuil Métier...")
        y_proba = final_model.predict_proba(X_val_top_arr)[:, 1]
        
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        min_cost = float('inf')
        
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred_t).ravel()
            cost = COST_FN * fn + COST_FP * fp
            if cost < min_cost:
                min_cost = cost
                best_threshold = t
                
        # Calcul des métriques avec le seuil optimal
        y_pred_best = (y_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_best).ravel()
        
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred_best)
        precision = precision_score(y_val, y_pred_best)
        recall = recall_score(y_val, y_pred_best)
        
        # Logs vers MLflow
        mlflow.log_metric("training_time_seconds", train_time)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("min_business_cost", min_cost)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        
        # Log du modèle dans MLflow
        mlflow.lightgbm.log_model(final_model, "lightgbm-model")
        
        print("--- RÉSULTATS MLOPS ---")
        print(f"Seuil optimal (FN=10, FP=1) : {best_threshold:.3f}")
        print(f"Coût Métier Minimum : {min_cost}")
        print(f"ROC AUC : {auc:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"Recall (Sensibilité) : {recall:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Matrice de confusion : TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # 9. Sauvegarde Exhaustive
    print(f"\nSauvegarde du modèle et des métriques dans {MODEL_PATH}")
    export_dict = {
        'model': final_model,
        'imputer': final_imputer,
        'features': top_features,
        'metrics': {
            'best_threshold': best_threshold,
            'min_business_cost': min_cost,
            'roc_auc': auc,
            'f1_score': f1,
            'recall': recall,
            'precision': precision,
            'training_time_seconds': train_time
        }
    }
    joblib.dump(export_dict, MODEL_PATH)
    print("✅ Pipeline Phase 0 terminé avec succès !")

if __name__ == "__main__":
    main()
