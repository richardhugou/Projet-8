import pandas as pd
import json
import os

from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
from evidently.presets import DataDriftPreset

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, "logs", "production_inference.jsonl")
REPORT_DIR = os.path.join(BASE_DIR, "monitoring", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def run_drift_analysis():
    print("--- Analyse de Drift (Evidently v0.7.x Officiel) ---")

    if not os.path.exists(LOG_FILE):
        print(f"❌ Erreur : Fichier logs absent {LOG_FILE}")
        return

    # 1. Chargement des logs
    data = []
    with open(LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry["inputs"])

    df = pd.DataFrame(data)

    # 2. Splitting 400x3 (Référence vs Production)
    df_ref = df.iloc[:400].copy()
    df_curr = df.iloc[400:].copy()

    # 3. Définition du Schéma pour forcer les types numériques
    schema = DataDefinition(numerical_columns=df.columns.tolist())

    # 4. Conversion en Datasets Evidently
    dataset_curr = Dataset.from_pandas(df_curr, data_definition=schema)
    dataset_ref = Dataset.from_pandas(df_ref, data_definition=schema)

    # 5. Création et Exécution du Rapport
    report = Report([DataDriftPreset()])
    my_eval = report.run(dataset_curr, dataset_ref)

    # 6. Exportation (JSON + HTML)
    json_path = os.path.join(REPORT_DIR, "drift_report.json")
    with open(json_path, "w") as f:
        json.dump(my_eval.dict(), f)

    html_path = os.path.join(REPORT_DIR, "drift_report.html")
    my_eval.save_html(html_path)

    print("✅ Analyse terminée avec succès.")
    print(f"Rapport JSON : {json_path}")
    print(f"Rapport HTML : {html_path}")


if __name__ == "__main__":
    run_drift_analysis()
