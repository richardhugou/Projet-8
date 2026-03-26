import pandas as pd
import json
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import *

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'production_inference.jsonl')
REPORT_DIR = os.path.join(BASE_DIR, 'monitoring', 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def run_drift_analysis():
    print("--- Analyse de Data Drift (Evidently AI) ---")
    
    if not os.path.exists(LOG_FILE):
        print(f"Erreur : Fichier de logs absent {LOG_FILE}")
        return

    # 1. Chargement des données depuis les logs JSONL
    print(f"Chargement des logs depuis {LOG_FILE}...")
    data = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # On aplatit les inputs et on ajoute la prédiction
            row = entry['inputs'].copy()
            row['probability'] = entry['outputs']['probability_default']
            row['prediction'] = entry['outputs']['prediction']
            data.append(row)
            
    df = pd.DataFrame(data)
    print(f"Total logs chargés : {len(df)}")

    # 2. Séparation Référence (400 premières lignes) vs Actuel (800 suivantes)
    # Rappel du script de simulation : 
    # [0:400]   -> Train based (Reference)
    # [400:800]  -> Test based (Production nominale)
    # [800:1200] -> Test based avec Drift Salaire
    
    reference_data = df.iloc[:400]
    current_data = df.iloc[400:]

    # 3. Création du Rapport Evidently
    print("Génération du rapport de Drift...")
    drift_report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset()
    ])

    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # 4. Exportation des résultats
    html_path = os.path.join(REPORT_DIR, 'drift_report.html')
    json_path = os.path.join(REPORT_DIR, 'drift_report.json')
    
    drift_report.save_html(html_path)
    drift_report.save_json(json_path)
    
    print(f"✅ Analyse terminée.")
    print(f"Rapport HTML : {html_path}")
    print(f"Rapport JSON : {json_path}")

if __name__ == "__main__":
    run_drift_analysis()
