# AegisPay // Credit Card Fraud Detection using Anomaly Detection
> **Semester-5 Artificial Intelligence Mini-Project**  
> *Dataset: Kaggle Credit Card Fraud Detection (ULB Benchmark)*  
> *Primary Algorithm: Isolation Forest (Unsupervised) | Benchmark: Random Forest (Supervised)*

---

## 📌 Executive Summary
Credit card fraud is an asymmetrical, moving-target problem: out of **284,807 transactions**, only **492 are fraudulent (0.172%)**. Traditional supervised classifiers overfit to historical fraud patterns and completely fail to detect **zero-day fraud exploits**. 

This project implements **AegisPay**, an end-to-end fraud detection system powered by **Unsupervised Anomaly Detection (Isolation Forest)** paired with a modern, cyber-fintech dashboard. It offers:
1. **Real-time transaction risk scoring & classification** (Legitimate, Suspicious, Fraudulent).
2. **Explainability attribution** (Z-score deviation identifying which PCA feature drove the anomaly).
3. **1-Click testing** using real Kaggle transactions.
4. **Batch CSV auditor** for high-throughput compliance scanning.
5. **Academic benchmark dashboard** comparing Unsupervised Isolation Forest vs Supervised Baseline.

---

## 🏗️ System Architecture

```
[ Kaggle Dataset (creditcard.csv) ] 
               │
               ▼
   [ Preprocessing & Scaling ]
   - RobustScaler on Amount & Time
   - Reference Distribution Profiling (μ, σ per feature)
   - Stratified Train/Test Split (80/20)
               │
       ┌───────┴───────┐
       ▼               ▼
[ Isolation Forest ] [ Supervised RF ]
(Unsupervised)       (Baseline Benchmark)
       │               │
       └───────┬───────┘
               ▼
[ Serialized Models (.joblib) & Reference Profiles ]
               │
               ▼
   [ Flask REST API Service ]
   ├── POST /api/predict (Single Txn Scoring & Explainability)
   ├── POST /api/predict/batch (CSV File Vectorized Audit)
   ├── GET  /api/samples (Pre-loaded Real Test Profiles)
   └── GET  /api/metrics (Benchmark Data & Confusion Matrices)
               │
               ▼
[ Modern Cyber-Fintech Glassmorphism Dashboard ]
   ├── Live Circular SVG Risk Gauge (0 - 100%)
   ├── 1-Click Kaggle Benchmark Test Cards
   ├── Top Anomaly Driver Z-Score Breakdown
   └── Interactive Model Comparison & Confusion Matrix Heatmaps
```

---

## 🔬 Mathematical Methodology

### 1. Isolation Forest (`IsolationForest`)
Rather than modeling dense normal regions (which suffers from the *Curse of Dimensionality* in 30 features), Isolation Forest explicitly isolates outliers using an ensemble of randomized trees.
* **Intuition:** Because fraudulent transactions are rare and have extreme PCA coordinates ($V_{14}, V_{12}$ drop), they require significantly **shorter path lengths $h(x)$** to isolate than normal transactions.
* **Anomaly Score Formula:**
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
where $E(h(x))$ is the average path length across all isolation trees, and $c(n)$ is the average path length of an unsuccessful search in a Binary Search Tree (BST):
$$c(n) = 2 \ln(n - 1) + 0.5772156649\ (\text{Euler-Mascheroni constant}) - \frac{2(n - 1)}{n}$$
* **Score Interpretation:**
  - If $s \to 1$: Highly anomalous (Fraud alert).
  - If $s < 0.5$: Well within the normal transaction cluster.

### 2. Feature Scaling (`RobustScaler`)
Standard scalers calculate mean and variance, which are drastically skewed by multi-thousand-dollar transactions. We use `RobustScaler`:
$$x_{\text{scaled}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$
where $\text{IQR} = Q_3 - Q_1$.

### 3. Real-Time Explainability (Z-Score Attribution)
For any transaction $x$, each feature $i$ is compared against the normal cardholder distribution:
$$z_i = \frac{x_i - \mu_{\text{normal}, i}}{\sigma_{\text{normal}, i}}$$
The features with the highest $|z_i|$ are presented on the dashboard as the primary anomaly drivers (e.g., $V_{14}$ drop).

---

## 📊 Benchmark Results (56,962 Holdout Test Set)

| Metric | Isolation Forest *(Unsupervised)* | Random Forest *(Supervised Baseline)* |
| :--- | :--- | :--- |
| **Paradigm** | Unsupervised (Tree Partitioning) | Supervised (Balanced Depth 10) |
| **ROC-AUC Score** | **0.9546** | 0.9798 |
| **PR-AUC (Avg Precision)** | 0.1475 | **0.8321** |
| **Holdout Recall (Fraud)** | 25.5% (25/98 caught) | **85.7%** (84/98 caught) |
| **Inference Latency** | **0.005 ms / transaction** | 0.003 ms / transaction |
| **Training Time** | **2.07 seconds** | 14.32 seconds |
| **Zero-Day Fraud Resilience** | ⭐⭐⭐⭐⭐ **High (No labels required)** | ⭐⭐ Low (Requires prior training labels) |

---

## 🚀 How to Run the Project

### 1. Requirements
* Python 3.10+
* Packages: `scikit-learn`, `pandas`, `numpy`, `flask`, `flask-cors`, `joblib`

### 2. Quickstart (One Command)
Run the root runner:
```bash
py run.py
```
Open your browser at: **`http://127.0.0.1:5000`**

### 3. Manual Step-by-Step
```bash
# 1. Preprocess dataset & extract samples
py src/preprocess.py

# 2. Train models and generate benchmark metrics
py src/train.py

# 3. Start dashboard server
py src/app.py
```

---

## 🎓 Viva & Evaluation Questions (Semester-5 AI)

### Q1: Why use Anomaly Detection instead of standard Supervised Learning?
> **Answer:** In real-world banking, fraud labels suffer from a **60–90 day latency** (chargebacks/disputes) and fraud patterns evolve constantly. Supervised models only detect historical attacks they were trained on. **Isolation Forest** detects novel, zero-day fraud tactics because it flags any statistical deviation from legitimate spending behavior, without requiring fraud labels.

### Q2: Why is Accuracy a misleading metric for this dataset?
> **Answer:** The dataset contains 99.827% legitimate transactions. A naive model that predicts every single transaction as "Legitimate" would achieve **99.83% accuracy**, but catch 0% of fraud. Therefore, **PR-AUC (Precision-Recall AUC)**, **Recall**, and **Cost-Matrix loss** are the correct evaluation metrics.

### Q3: What do the features V1 to V28 represent?
> **Answer:** For customer privacy and PCI-DSS compliance, the original transaction features (cardholder location, merchant ID, device IP, terminal type) were transformed using **Principal Component Analysis (PCA)**. `Time` and `Amount` were left un-transformed.

### Q4: Why did we scale Amount using `RobustScaler` instead of `StandardScaler`?
> **Answer:** `Amount` contains extreme outliers (ranging from \$0 to \$25,691). `StandardScaler` uses the mean and variance, which are vulnerable to extreme outliers. `RobustScaler` uses the **median and Interquartile Range (IQR)**, ensuring robust scaling unaffected by huge spend spikes.
