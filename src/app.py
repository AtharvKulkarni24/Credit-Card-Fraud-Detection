import json
import logging
import io
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd

from detector import FraudDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
SAMPLES_PATH = BASE_DIR / "data" / "sample_transactions.json"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
CORS(app)

# Initialize Detector Engine
detector = FraudDetector()

# In-memory storage for latest batch export
latest_batch_results = None


@app.route("/")
def index():
    """Serves the main dashboard frontend."""
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "Credit Card Fraud Anomaly Engine"})


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns curated realistic sample transactions for 1-click UI testing."""
    if not SAMPLES_PATH.exists():
        return jsonify({"error": "Samples file not found."}), 404
        
    with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return jsonify(samples)


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns model benchmark comparison, ROC curve, and confusion matrices."""
    if not METRICS_PATH.exists():
        return jsonify({"error": "Metrics file not found. Run train.py first."}), 404
        
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return jsonify(metrics)


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """Evaluates a single transaction."""
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No transaction payload provided."}), 400
            
        features = data.get("features", data)
        result = detector.predict_transaction(features)
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error predicting transaction: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/batch", methods=["POST"])
def predict_batch_csv():
    """Accepts an uploaded CSV file, scores each transaction, and returns audit summary."""
    global latest_batch_results
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Key 'file' is required."}), 400
        
    file = request.files["file"]
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Invalid file type. Please upload a .csv file."}), 400
        
    try:
        df = pd.read_csv(file)
        # Limit batch demo to max 10,000 rows for UI responsiveness
        if len(df) > 10000:
            df = df.iloc[:10000]
            
        scored_df, summary = detector.predict_batch(df)
        latest_batch_results = scored_df
        
        # Extract top 20 flagged frauds for preview table
        flagged_preview = scored_df[scored_df["Prediction"] != "LEGITIMATE"].head(20).to_dict(orient="records")
        
        return jsonify({
            "summary": summary,
            "flagged_preview": flagged_preview,
            "processed_count": len(scored_df)
        })
    except Exception as e:
        logging.error(f"Error in batch processing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/export/batch", methods=["GET"])
def export_batch_csv():
    """Downloads the latest audited batch results with risk predictions."""
    global latest_batch_results
    if latest_batch_results is None:
        return jsonify({"error": "No audited batch found. Upload a file first."}), 400
        
    output = io.BytesIO()
    latest_batch_results.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="fraud_audit_results.csv"
    )


@app.route("/api/sample-batch/download", methods=["GET"])
def download_sample_batch():
    """Provides a sample 100-transaction CSV file for instant batch testing."""
    sample_path = BASE_DIR / "test_batch_sample.csv"
    return send_file(
        str(sample_path),
        mimetype="text/csv",
        as_attachment=True,
        download_name="test_batch_sample.csv"
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print(">> AegisPay Fraud Detection Engine & Dashboard Online!")
    print(">> Open Browser at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
