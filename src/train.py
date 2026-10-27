import json
import logging
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from preprocess import load_and_inspect_data, preprocess_and_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IF_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
RF_MODEL_PATH = MODELS_DIR / "baseline_rf.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"


def evaluate_model(name, y_true, y_pred, y_scores, latency_ms):
    """Computes academic-standard evaluation metrics for imbalanced fraud data."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Calculate curve points for dashboard visualization (downsampled to ~30 points)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    
    step_roc = max(1, len(fpr) // 30)
    step_pr = max(1, len(precisions) // 30)
    
    roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr[::step_roc], tpr[::step_roc])]
    pr_points = [{"recall": round(float(r), 4), "precision": round(float(p), 4)} for r, p in zip(recalls[::step_pr], precisions[::step_pr])]

    metrics = {
        "model_name": name,
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "latency_ms": round(float(latency_ms), 3),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "curves": {
            "roc": roc_points,
            "pr": pr_points
        }
    }
    
    logging.info(f"--- {name} Results ---")
    logging.info(f"PR-AUC (Avg Precision): {metrics['pr_auc']:.4f}")
    logging.info(f"ROC-AUC:               {metrics['roc_auc']:.4f}")
    logging.info(f"Recall (Fraud caught): {metrics['recall']:.4f} ({tp}/{tp+fn})")
    logging.info(f"Precision:             {metrics['precision']:.4f} ({tp}/{tp+fp})")
    logging.info(f"F1-Score:              {metrics['f1_score']:.4f}")
    logging.info(f"Inference Latency:     {metrics['latency_ms']:.3f} ms/txn")
    
    return metrics


def train_models():
    """Trains Isolation Forest and Baseline Random Forest, saving models and metrics."""
    df = load_and_inspect_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(df)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    all_metrics = {}

    # =========================================================================
    # 1. Primary Model: Isolation Forest (Unsupervised Anomaly Detection)
    # =========================================================================
    logging.info("\n>>> Training Primary Model: Isolation Forest (Unsupervised)...")
    contamination = float(y_train.sum() / len(y_train))  # ~0.0017
    logging.info(f"Tuning contamination parameter to empirical rate: {contamination:.5f}")
    
    iso_forest = IsolationForest(
        n_estimators=150,
        max_samples=256,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    start_train = time.time()
    iso_forest.fit(X_train)
    logging.info(f"Isolation Forest trained in {time.time() - start_train:.2f} seconds.")

    # Isolation Forest Inference & Evaluation
    start_inf = time.time()
    # decision_function gives signed distance: lower / negative means anomaly
    raw_scores = iso_forest.decision_function(X_test)
    inf_time = (time.time() - start_inf) / len(X_test) * 1000  # ms per txn
    
    # Invert scores so that higher score = higher anomaly / fraud probability
    # Offset and scale raw_scores to [0, 1] range
    anomaly_scores = -raw_scores
    anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
    
    # IsolationForest: -1 for anomaly, 1 for normal
    raw_preds = iso_forest.predict(X_test)
    y_pred_if = np.where(raw_preds == -1, 1, 0)
    
    if_metrics = evaluate_model(
        name="Isolation Forest (Unsupervised)",
        y_true=y_test,
        y_pred=y_pred_if,
        y_scores=anomaly_scores_norm,
        latency_ms=inf_time
    )
    all_metrics["isolation_forest"] = if_metrics
    
    joblib.dump(iso_forest, IF_MODEL_PATH)
    logging.info(f"Saved Isolation Forest to: {IF_MODEL_PATH}")

    # =========================================================================
    # 2. Benchmark Model: Supervised Random Forest (Balanced)
    # =========================================================================
    logging.info("\n>>> Training Benchmark Model: Random Forest (Supervised Baseline)...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    start_train = time.time()
    rf.fit(X_train, y_train)
    logging.info(f"Random Forest trained in {time.time() - start_train:.2f} seconds.")

    start_inf = time.time()
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = rf.predict(X_test)
    inf_time_rf = (time.time() - start_inf) / len(X_test) * 1000
    
    rf_metrics = evaluate_model(
        name="Random Forest (Supervised Baseline)",
        y_true=y_test,
        y_pred=rf_preds,
        y_scores=rf_probs,
        latency_ms=inf_time_rf
    )
    all_metrics["random_forest"] = rf_metrics
    
    joblib.dump(rf, RF_MODEL_PATH)
    logging.info(f"Saved Random Forest to: {RF_MODEL_PATH}")

    # Save summary metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    logging.info(f"\nSaved all benchmark metrics to: {METRICS_PATH}")
    
    return all_metrics


if __name__ == "__main__":
    train_models()
    print("\n[SUCCESS] Phase 3: Model Training & Evaluation completed successfully!")
