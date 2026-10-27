import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IF_MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"
RF_MODEL_PATH = MODELS_DIR / "baseline_rf.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
PROFILE_PATH = MODELS_DIR / "reference_profile.json"


class FraudDetector:
    """Production Inference & Explainability Engine for Fraud Detection."""

    def __init__(self):
        self.load_artifacts()

    def load_artifacts(self):
        logging.info("Loading ML models and preprocessing artifacts...")
        if not IF_MODEL_PATH.exists() or not SCALER_PATH.exists():
            raise FileNotFoundError("Models or scaler not found. Run train.py first.")
            
        self.iso_forest = joblib.load(IF_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.rf_model = joblib.load(RF_MODEL_PATH) if RF_MODEL_PATH.exists() else None
        
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            self.reference_profile = json.load(f)
            
        self.feature_order = [f"V{i}" for i in range(1, 29)] + ["scaled_time", "scaled_amount"]

    def _prepare_vector(self, txn_dict):
        """Preprocesses input transaction dictionary into scaled feature DataFrame."""
        time_val = float(txn_dict.get("Time", 0.0))
        amount_val = float(txn_dict.get("Amount", 0.0))
        
        # Scale Time and Amount with fitted RobustScaler
        ta_df = pd.DataFrame([[time_val, amount_val]], columns=["Time", "Amount"])
        scaled_time_amt = self.scaler.transform(ta_df)
        scaled_time = float(scaled_time_amt[0, 0])
        scaled_amount = float(scaled_time_amt[0, 1])
        
        # Build ordered row matching X_train column order
        row_dict = {}
        for i in range(1, 29):
            col = f"V{i}"
            row_dict[col] = float(txn_dict.get(col, 0.0))
        row_dict["scaled_time"] = scaled_time
        row_dict["scaled_amount"] = scaled_amount
        
        X_df = pd.DataFrame([row_dict])
        return X_df, time_val, amount_val, scaled_time, scaled_amount

    def compute_feature_explainability(self, txn_dict, top_k=5):
        """
        Computes z-score deviations against legitimate transaction baseline.
        Identifies top contributing anomalous drivers.
        """
        deviations = []
        
        for feat, stats in self.reference_profile.items():
            val = float(txn_dict.get(feat, 0.0))
            mean = stats["mean"]
            std = stats["std"]
            z_score = (val - mean) / std
            
            deviations.append({
                "feature": feat,
                "value": round(val, 4),
                "normal_mean": round(mean, 4),
                "z_score": round(float(z_score), 2),
                "abs_z": abs(float(z_score))
            })
            
        # Sort by absolute deviation
        deviations.sort(key=lambda x: x["abs_z"], reverse=True)
        top_drivers = deviations[:top_k]
        
        # Annotate context for each top driver
        for item in top_drivers:
            feat = item["feature"]
            z = item["z_score"]
            if feat in ["V14", "V12", "V10", "V17"]:
                desc = "Critical latent PCA security component showing severe anomaly pattern"
            elif feat == "Amount":
                desc = "Extreme deviation in transaction size compared to median baseline"
            elif feat == "Time":
                desc = "Unusual transaction timing offset relative to standard cardholder window"
            else:
                desc = "Significant statistical divergence from legitimate spending profile"
                
            item["direction"] = "Spike Above Baseline" if z > 0 else "Drop Below Baseline"
            item["description"] = desc
            
        return top_drivers

    def predict_transaction(self, txn_dict):
        """
        Runs dual-engine scoring (Isolation Forest Anomaly Score + RF Supervised Probability).
        Returns risk tier, explanation, and calibrated metrics.
        """
        X_vec, raw_time, raw_amount, s_time, s_amount = self._prepare_vector(txn_dict)
        
        # 1. Isolation Forest Anomaly Scoring
        raw_decision = float(self.iso_forest.decision_function(X_vec)[0])
        # Decision function: typical normal is > 0.05, anomaly is < 0.
        # We map to 0-100% anomaly score calibrated to the Kaggle distribution
        # Normal transactions center around ~0.15; extreme frauds go down to -0.35
        # Calibrated formula:
        anomaly_score = max(0.0, min(100.0, (0.12 - raw_decision) / 0.35 * 100.0))
        is_if_anomaly = bool(self.iso_forest.predict(X_vec)[0] == -1)
        
        # 2. Random Forest Supervised Probability
        rf_prob = 0.0
        if self.rf_model:
            rf_prob = float(self.rf_model.predict_proba(X_vec)[0, 1]) * 100.0
            
        # 3. Blended Composite Risk Score & Status
        composite_risk = (0.55 * anomaly_score) + (0.45 * rf_prob)
        
        if composite_risk >= 65.0 or (is_if_anomaly and rf_prob >= 40.0):
            risk_level = "CRITICAL"
            status = "FRAUDULENT"
            action = "Transaction Blocked: Stolen credential signature detected"
            badge_color = "#ef4444"  # Neon Red
        elif composite_risk >= 35.0 or is_if_anomaly:
            risk_level = "ELEVATED"
            status = "SUSPICIOUS"
            action = "Held for Review: Requires secondary 2FA / SMS challenge"
            badge_color = "#f59e0b"  # Amber
        else:
            risk_level = "LOW"
            status = "LEGITIMATE"
            action = "Instant Approval: Consistent with cardholder behavioral profile"
            badge_color = "#10b981"  # Emerald

        # 4. Explainability
        top_drivers = self.compute_feature_explainability(txn_dict, top_k=5)

        return {
            "status": status,
            "risk_level": risk_level,
            "action": action,
            "badge_color": badge_color,
            "composite_risk_score": round(composite_risk, 1),
            "anomaly_score": round(anomaly_score, 1),
            "supervised_probability": round(rf_prob, 1),
            "is_anomaly": is_if_anomaly,
            "raw_decision_score": round(raw_decision, 4),
            "transaction_amount": raw_amount,
            "top_drivers": top_drivers
        }

    def predict_batch(self, df):
        """Scores an uploaded CSV DataFrame in a vectorized manner."""
        required_cols = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Uploaded CSV is missing required columns: {missing[:5]}")
            
        scaled_ta = self.scaler.transform(df[["Time", "Amount"]])
        
        X_batch = df[[f"V{i}" for i in range(1, 29)]].copy()
        X_batch["scaled_time"] = scaled_ta[:, 0]
        X_batch["scaled_amount"] = scaled_ta[:, 1]
        
        # Batch inference
        if_decisions = self.iso_forest.decision_function(X_batch)
        anomaly_scores = np.clip((0.12 - if_decisions) / 0.35 * 100.0, 0.0, 100.0)
        
        rf_probs = np.zeros(len(df))
        if self.rf_model:
            rf_probs = self.rf_model.predict_proba(X_batch)[:, 1] * 100.0
            
        composite = (0.55 * anomaly_scores) + (0.45 * rf_probs)
        
        # Categorize
        statuses = []
        for score, if_dec, rfp in zip(composite, if_decisions, rf_probs):
            if score >= 65.0 or (if_dec < 0 and rfp >= 40.0):
                statuses.append("FRAUD")
            elif score >= 35.0 or if_dec < 0:
                statuses.append("SUSPICIOUS")
            else:
                statuses.append("LEGITIMATE")
                
        results_df = df.copy()
        results_df["Anomaly_Score"] = np.round(anomaly_scores, 1)
        results_df["Supervised_Prob"] = np.round(rf_probs, 1)
        results_df["Risk_Score"] = np.round(composite, 1)
        results_df["Prediction"] = statuses
        
        summary = {
            "total_transactions": len(df),
            "fraud_count": int((results_df["Prediction"] == "FRAUD").sum()),
            "suspicious_count": int((results_df["Prediction"] == "SUSPICIOUS").sum()),
            "legitimate_count": int((results_df["Prediction"] == "LEGITIMATE").sum()),
            "total_amount_audited": round(float(df["Amount"].sum()), 2),
            "fraud_amount_prevented": round(float(df.loc[results_df["Prediction"] == "FRAUD", "Amount"].sum()), 2)
        }
        
        return results_df, summary


if __name__ == "__main__":
    detector = FraudDetector()
    with open(BASE_DIR / "data" / "sample_transactions.json", "r") as f:
        samples = json.load(f)
    print("\n[TEST] Testing 1 Legitimate and 1 Fraud sample:")
    res_norm = detector.predict_transaction(samples[0]["features"])
    print("Normal Sample Result:", res_norm["status"], f"Risk: {res_norm['composite_risk_score']}%")
    res_fraud = detector.predict_transaction(samples[3]["features"])
    print("Fraud Sample Result:", res_fraud["status"], f"Risk: {res_fraud['composite_risk_score']}%")
