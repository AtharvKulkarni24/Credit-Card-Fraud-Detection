import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODELS_DIR = BASE_DIR / "models"
SAMPLES_PATH = BASE_DIR / "data" / "sample_transactions.json"
PROFILE_PATH = MODELS_DIR / "reference_profile.json"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


def load_and_inspect_data(filepath=DATA_PATH):
    """Loads dataset and prints basic distribution stats."""
    logging.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    total = len(df)
    fraud_count = int(df["Class"].sum())
    normal_count = total - fraud_count
    fraud_pct = (fraud_count / total) * 100

    logging.info(f"Total transactions: {total:,}")
    logging.info(f"Normal transactions (Class 0): {normal_count:,} ({100 - fraud_pct:.3f}%)")
    logging.info(f"Fraud transactions  (Class 1): {fraud_count:,} ({fraud_pct:.3f}%)")
    logging.info(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def extract_curated_samples(df, num_normal=3, num_fraud=3):
    """
    Extracts realistic sample transactions (normal & fraud) with descriptive scenarios
    for 1-click testing in the frontend dashboard.
    """
    logging.info("Extracting curated sample test cases for UI...")
    normal_df = df[df["Class"] == 0]
    fraud_df = df[df["Class"] == 1]
    
    samples = []
    
    # 1. Normal Small Everyday purchase
    small_normal = normal_df[(normal_df["Amount"] > 10) & (normal_df["Amount"] < 50)].iloc[10]
    samples.append({
        "id": "sample-norm-1",
        "title": "Everyday Grocery Purchase",
        "scenario": "Typical daytime grocery store transaction with normal PCA feature bounds.",
        "ground_truth": "Legitimate",
        "is_fraud": 0,
        "amount": float(small_normal["Amount"]),
        "features": {col: float(small_normal[col]) for col in df.columns if col != "Class"}
    })

    # 2. Normal Recurring / Moderate purchase
    mod_normal = normal_df[(normal_df["Amount"] >= 100) & (normal_df["Amount"] <= 300)].iloc[25]
    samples.append({
        "id": "sample-norm-2",
        "title": "Electronics Retail Store",
        "scenario": "Standard retail electronics purchase; moderate amount with normal spending profile.",
        "ground_truth": "Legitimate",
        "is_fraud": 0,
        "amount": float(mod_normal["Amount"]),
        "features": {col: float(mod_normal[col]) for col in df.columns if col != "Class"}
    })

    # 3. High-Amount Legitimate purchase (Stress test for False Positives)
    high_normal = normal_df[normal_df["Amount"] > 1500].iloc[5]
    samples.append({
        "id": "sample-norm-3",
        "title": "High-Value Luxury Booking",
        "scenario": "Legitimate high-ticket travel/hotel transaction; high amount but regular feature vector.",
        "ground_truth": "Legitimate",
        "is_fraud": 0,
        "amount": float(high_normal["Amount"]),
        "features": {col: float(high_normal[col]) for col in df.columns if col != "Class"}
    })

    # 4. Confirmed High-Confidence Fraud (Aggressive anomaly in V14, V12, V10)
    fraud_1 = fraud_df.sort_values(by="V14", ascending=True).iloc[0]
    samples.append({
        "id": "sample-fraud-1",
        "title": "Compromised Account Draining",
        "scenario": "High-confidence anomaly with massive deviations in latent security features V14 and V12.",
        "ground_truth": "Fraudulent",
        "is_fraud": 1,
        "amount": float(fraud_1["Amount"]),
        "features": {col: float(fraud_1[col]) for col in df.columns if col != "Class"}
    })

    # 5. Nighttime Low-Amount Fraud (Card testing attack)
    fraud_2 = fraud_df[fraud_df["Amount"] < 5.0].iloc[2]
    samples.append({
        "id": "sample-fraud-2",
        "title": "Micro-Charge Probe Attack",
        "scenario": "Low amount test transaction used by fraudsters to verify stolen credentials before big drain.",
        "ground_truth": "Fraudulent",
        "is_fraud": 1,
        "amount": float(fraud_2["Amount"]),
        "features": {col: float(fraud_2[col]) for col in df.columns if col != "Class"}
    })

    # 6. Borderline / Subtle Fraud
    fraud_3 = fraud_df[(fraud_df["Amount"] >= 50) & (fraud_df["Amount"] <= 200)].iloc[8]
    samples.append({
        "id": "sample-fraud-3",
        "title": "Subtle E-Commerce Impersonation",
        "scenario": "Mid-tier transaction where features moderately deviate from cardholder cluster.",
        "ground_truth": "Fraudulent",
        "is_fraud": 1,
        "amount": float(fraud_3["Amount"]),
        "features": {col: float(fraud_3[col]) for col in df.columns if col != "Class"}
    })

    # Save to JSON
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    logging.info(f"Saved {len(samples)} curated sample test cases to: {SAMPLES_PATH}")


def compute_reference_profile(normal_df):
    """
    Computes mean and std of all features on legitimate transactions.
    Used for real-time explainability (z-score feature attribution).
    """
    logging.info("Computing reference distribution from legitimate transactions...")
    feature_cols = [c for c in normal_df.columns if c != "Class"]
    
    profile = {}
    for col in feature_cols:
        profile[col] = {
            "mean": float(normal_df[col].mean()),
            "std": float(normal_df[col].std()) if float(normal_df[col].std()) > 1e-6 else 1.0,
            "median": float(normal_df[col].median()),
            "q25": float(normal_df[col].quantile(0.25)),
            "q75": float(normal_df[col].quantile(0.75))
        }
        
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    logging.info(f"Saved feature reference profile to: {PROFILE_PATH}")
    return profile


def preprocess_and_split(df, test_size=0.2, random_state=42):
    """
    Scales Amount & Time with RobustScaler, keeps PCA features intact,
    and returns stratified train/test partitions.
    """
    logging.info("Preprocessing features with RobustScaler on Amount & Time...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Compute reference profile on normal transactions for explainability
    normal_df = df[df["Class"] == 0]
    compute_reference_profile(normal_df)
    
    # 2. Extract curated samples
    extract_curated_samples(df)
    
    # 3. Fit RobustScaler on Time and Amount
    scaler = RobustScaler()
    scaled_time_amount = scaler.fit_transform(df[["Time", "Amount"]])
    
    df_scaled = df.copy()
    df_scaled["scaled_time"] = scaled_time_amount[:, 0]
    df_scaled["scaled_amount"] = scaled_time_amount[:, 1]
    
    # Drop original unscaled Time and Amount
    df_scaled.drop(columns=["Time", "Amount"], inplace=True)
    
    # Save the fitted scaler
    joblib.dump(scaler, SCALER_PATH)
    logging.info(f"Saved fitted RobustScaler to: {SCALER_PATH}")
    
    # 4. Feature and Target Separation
    X = df_scaled.drop(columns=["Class"])
    y = df_scaled["Class"]
    
    # 5. Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    logging.info(f"Train split: {X_train.shape[0]:,} records ({int(y_train.sum())} frauds)")
    logging.info(f"Test split:  {X_test.shape[0]:,} records ({int(y_test.sum())} frauds)")
    
    return X_train, X_test, y_train, y_test, list(X.columns)


if __name__ == "__main__":
    df = load_and_inspect_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(df)
    print("\n[SUCCESS] Phase 2: Preprocessing and Sample Extraction completed successfully!")
