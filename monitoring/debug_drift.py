import pandas as pd
import json
import os

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'production_inference.jsonl')
REPORT_DIR = os.path.join(BASE_DIR, 'monitoring', 'reports')
os.makedirs(REPORT_DIR, exist_ok=True)

def run_drift_analysis():
    print("--- Analyse de Drift (Evidently v0.7.x Officiel) ---")
    
    data = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry['inputs'])
            
    df = pd.DataFrame(data)
    
    df_ref = df.iloc[:400].copy()
    df_curr = df.iloc[400:].copy()

    schema = DataDefinition(
        numerical_columns=df.columns.tolist()
    )

    dataset_curr = Dataset.from_pandas(
        df_curr,
        data_definition=schema
    )
    
    dataset_ref = Dataset.from_pandas(
        df_ref,
        data_definition=schema
    )

    report = Report([DataDriftPreset()])
    
    my_eval = report.run(dataset_curr, dataset_ref)
    
    json_path = os.path.join(REPORT_DIR, 'drift_report.json')
    with open(json_path, 'w') as f:
        json.dump(my_eval.dict(), f)
        
    html_path = os.path.join(REPORT_DIR, 'drift_report.html')
    my_eval.save_html(html_path)
    
    print(f"Analyse terminée. Rapport JSON : {json_path}")
    print(f"Rapport HTML interactif généré : {html_path}")

if __name__ == "__main__":
    run_drift_analysis()