#!/usr/bin/env python3
"""Extract essential production model metrics and generate a business-cost heatmap."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def create_domain_features_strict(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate training-time feature engineering and clipping behavior."""
    out = df.copy()

    if "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)

    for col in ("AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"):
        if col in out.columns:
            upper_limit = out[col].quantile(0.99)
            out[col] = out[col].clip(upper=upper_limit)

    # Domain features used during model training.
    out["DAYS_EMPLOYED_PERCENT"] = out["DAYS_EMPLOYED"] / out["DAYS_BIRTH"]
    out["YEARS_EMPLOYED"] = out["DAYS_EMPLOYED"] / -365
    out["YEARS_BIRTH"] = out["DAYS_BIRTH"] / -365
    out["CREDIT_INCOME_PERCENT"] = out["AMT_CREDIT"] / out["AMT_INCOME_TOTAL"]
    out["ANNUITY_INCOME_PERCENT"] = out["AMT_ANNUITY"] / out["AMT_INCOME_TOTAL"]
    out["CREDIT_TERM"] = out["AMT_ANNUITY"] / out["AMT_CREDIT"]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="model/optuna_scoring_model.joblib",
        help="Path to the production model artifact.",
    )
    parser.add_argument(
        "--data-path",
        default=(
            "internal_doc/Projet 6/Projet+Mise+en+prod+-+home-credit-default-risk/"
            "application_train.csv"
        ),
        help="Path to the labeled reference dataset used for certification.",
    )
    parser.add_argument(
        "--output-json",
        default="monitoring/reports/production_metrics_summary.json",
        help="Where to write the extracted metrics summary.",
    )
    parser.add_argument(
        "--heatmap-path",
        default="monitoring/reports/business_cost_heatmap.png",
        help="Where to save the business-cost heatmap.",
    )
    parser.add_argument(
        "--confusion-heatmap-path",
        default="monitoring/reports/confusion_matrix_heatmap.png",
        help="Where to save the confusion-matrix heatmap (raw counts).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cost-fn", type=float, default=10.0)
    parser.add_argument("--cost-fp", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("The model artifact must be a dictionary.")

    required = {"model", "imputer", "features", "metrics"}
    missing = sorted(required.difference(artifact.keys()))
    if missing:
        raise KeyError(f"Missing required artifact keys: {missing}")

    metrics = artifact.get("metrics", {})
    threshold = float(metrics.get("best_threshold", 0.5))

    df = pd.read_csv(data_path)
    df = create_domain_features_strict(df)

    y = df["TARGET"]
    x = df.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")

    cat_columns = x.select_dtypes(include=["object"]).columns.tolist()
    if cat_columns:
        x = pd.get_dummies(x, columns=cat_columns, drop_first=True)
    x = x.rename(columns=lambda col: re.sub(r"[^A-Za-z0-9_]+", "", col))

    _, x_val, _, y_val = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    selected_features = list(artifact["features"])
    x_val_top = x_val.reindex(columns=selected_features)
    x_val_imp = artifact["imputer"].transform(x_val_top)

    start = time.perf_counter()
    y_proba = artifact["model"].predict_proba(x_val_imp)[:, 1]
    inference_time_seconds = time.perf_counter() - start
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    business_cost = (fn * args.cost_fn) + (fp * args.cost_fp)
    auc_score = roc_auc_score(y_val, y_proba)
    report = classification_report(y_val, y_pred, output_dict=True)

    confusion_counts = np.array([[tn, fp], [fn, tp]], dtype=int)
    cost_matrix = np.array([[0, fp * args.cost_fp], [fn * args.cost_fn, 0]], dtype=float)

    confusion_heatmap_path = Path(args.confusion_heatmap_path)
    confusion_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    sns.heatmap(
        confusion_counts,
        annot=np.array(
            [
                [f"TN\n{tn}", f"FP\n{fp}"],
                [f"FN\n{fn}", f"TP\n{tp}"],
            ]
        ),
        fmt="",
        cmap="Blues",
        xticklabels=["Pred: Pret accorde (0)", "Pred: Pret refuse (1)"],
        yticklabels=["Reel: Solvable (0)", "Reel: Defaut (1)"],
    )
    plt.title("Matrice de confusion (comptages bruts)")
    plt.xlabel("Decision du modele")
    plt.ylabel("Etat reel")
    plt.tight_layout()
    plt.savefig(confusion_heatmap_path, dpi=150)
    plt.close()

    heatmap_path = Path(args.heatmap_path)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cost_matrix,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        xticklabels=["Pret accorde", "Pret refuse"],
        yticklabels=["Solvable", "Defaut"],
    )
    plt.title("Heatmap du cout metier")
    plt.xlabel("Decision du modele")
    plt.ylabel("Etat reel")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    summary = {
        "threshold": threshold,
        "training_time_seconds": float(metrics.get("training_time", np.nan)),
        "roc_auc": float(auc_score),
        "f1_score_class_1": float(report["1"]["f1-score"]),
        "recall_class_1": float(report["1"]["recall"]),
        "precision_class_1": float(report["1"]["precision"]),
        "f1_score_macro": float(report["macro avg"]["f1-score"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "TN": int(tn),
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "business_cost": float(business_cost),
        "inference_time_seconds": float(inference_time_seconds),
        "cost_fn": float(args.cost_fn),
        "cost_fp": float(args.cost_fp),
        "heatmap_path": str(heatmap_path),
        "confusion_heatmap_path": str(confusion_heatmap_path),
        "artifact_metrics": {
            "best_threshold": float(metrics.get("best_threshold", np.nan)),
            "training_time": float(metrics.get("training_time", np.nan)),
            "roc_auc": float(metrics.get("roc_auc", np.nan)),
            "f1_score": float(metrics.get("f1_score", np.nan)),
            "recall": float(metrics.get("recall", np.nan)),
            "precision": float(metrics.get("precision", np.nan)),
            "min_business_cost": float(metrics.get("min_business_cost", np.nan)),
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()