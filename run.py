"""
1-Click Runner for Credit Card Fraud Detection System.
Checks model artifacts and launches the Flask web dashboard on http://127.0.0.1:5000.
"""
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
IF_MODEL = MODELS_DIR / "isolation_forest.joblib"
SRC_DIR = BASE_DIR / "src"

def main():
    print("=" * 65)
    print("  CREDIT CARD FRAUD DETECTION // ANOMALY ENGINE")
    print("  Semester-5 AI Mini Project")
    print("=" * 65)

    # 1. Check if models are trained
    if not IF_MODEL.exists():
        print("\n[*] Pretrained models not found. Running training pipeline...")
        train_proc = subprocess.run([sys.executable, str(SRC_DIR / "train.py")])
        if train_proc.returncode != 0:
            print("[!] Training failed. Please check data/creditcard.csv.")
            return

    # 2. Launch Flask App
    print("\n[*] Launching Dashboard Server...")
    print("🌐 Dashboard URL: http://127.0.0.1:5000")
    print("Press Ctrl+C to terminate.")
    print("=" * 65 + "\n")
    
    subprocess.run([sys.executable, str(SRC_DIR / "app.py")])

if __name__ == "__main__":
    main()
