import os
import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET_CSV = DATA_DIR / "creditcard.csv"
DOWNLOADS_DIR = Path(os.path.expanduser("~")) / "Downloads"

def check_or_locate_dataset():
    """Checks if creditcard.csv exists in data/ or locates it in Downloads."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if TARGET_CSV.exists() and TARGET_CSV.stat().st_size > 10_000_000:
        print(f"[OK] Found dataset at: {TARGET_CSV} ({TARGET_CSV.stat().st_size / (1024*1024):.2f} MB)")
        return str(TARGET_CSV)

    print("[*] Checking Downloads folder for downloaded zip or csv...")
    # 1. Check for raw creditcard.csv in Downloads
    raw_in_downloads = DOWNLOADS_DIR / "creditcard.csv"
    if raw_in_downloads.exists():
        print(f"[*] Moving {raw_in_downloads} -> {TARGET_CSV}")
        shutil.copy(raw_in_downloads, TARGET_CSV)
        print("[OK] Dataset copied successfully!")
        return str(TARGET_CSV)

    # 2. Check for Kaggle archive.zip or creditcardfraud.zip in Downloads
    for zip_name in ["archive.zip", "creditcardfraud.zip", "creditcard.zip"]:
        zip_path = DOWNLOADS_DIR / zip_name
        if zip_path.exists():
            print(f"[*] Found zip file in Downloads: {zip_path}. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    if file_info.filename.endswith("creditcard.csv"):
                        file_info.filename = "creditcard.csv"
                        zip_ref.extract(file_info, DATA_DIR)
                        print(f"[OK] Extracted to {TARGET_CSV} successfully!")
                        return str(TARGET_CSV)

    print("[!] creditcard.csv not yet detected in data/ or Downloads.")
    print("    Please download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print(f"    and save or extract 'creditcard.csv' into: {DATA_DIR}")
    return None

if __name__ == "__main__":
    check_or_locate_dataset()
