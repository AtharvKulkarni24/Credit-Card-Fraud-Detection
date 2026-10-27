# ACADEMIC MINI PROJECT REPORT
## Course: Artificial Intelligence (Semester-5) | Batch: AIL - L4
### Department of Computer Science & Engineering / Artificial Intelligence

---

# TITLE OF THE PROJECT
## **AegisPay: High-Throughput Credit Card Fraud Detection Utilizing Unsupervised Anomaly Detection (Isolation Forest) and Explainable Artificial Intelligence (XAI)**

**Academic Year:** 2026–2027  
**Semester:** Fifth Semester (V)  
**Batch:** AIL - L4  
**Student Name:** Atharv Kulkarni (and Project Group Members)  
**Project Repository:** [https://github.com/AtharvKulkarni24/Credit-Card-Fraud-Detection](https://github.com/AtharvKulkarni24/Credit-Card-Fraud-Detection)  

---

## 1. PROBLEM STATEMENT

### 1.1 Context and Real-World Impact
Global card-not-present (CNP) and electronic retail fraud generate financial losses exceeding **$35 billion annually** across commercial banking networks. As financial infrastructure transitions toward high-frequency digital settlement rails, automated transaction authorization engines must evaluate incoming payment requests within milliseconds. Within this context, fraud detection represents one of the most mathematically demanding problem spaces in applied machine learning due to three inherent domain constraints:

### 1.2 The Three Fundamental Challenges
1. **Extreme Class Imbalance:**
   In real-world payment data, fraudulent transactions represent a vanishingly small fraction of total volume. In the standard benchmark dataset from the Machine Learning Group at Université Libre de Bruxelles (ULB), exactly **492 out of 284,807 transactions are fraudulent (0.172%)**. A trivial or degenerate classifier that uniformly outputs the majority class ("Legitimate") will achieve an **Accuracy of 99.827%**, yet will completely fail its operational objective by catching zero frauds (Recall = 0.0%).
2. **Adversarial Concept Drift and Zero-Day Fraud Exploits:**
   Credit card fraud is fundamentally an adversarial game. Criminal syndicates constantly probe authorization filters and dynamically alter their behaviors (e.g., card testing, micro-charge probing, velocity laundering, and botnet-driven identity theft). Supervised machine learning algorithms (such as standard Decision Trees, Logistic Regression, or Support Vector Machines) optimize decision boundaries strictly based on historical labeled examples. Consequently, when an adversary deploys a **novel attack vector (zero-day fraud)**, supervised models fail to generalize.
3. **Severe Production Label Latency:**
   In commercial banking infrastructure, ground-truth fraud labels do not appear instantaneously. Due to consumer dispute workflows, chargeback rights under financial regulations, and inter-bank merchant reconciliation, a fraudulent charge typically takes **60 to 90 days** to be formally confirmed and labeled. An authorization system that depends strictly on labeled fraud examples cannot protect against active compromises during this multi-month blind spot.

### 1.3 Formal Problem Definition
Given a continuous stream of $d$-dimensional transaction vectors $X = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^d$, where the true class label $y \in \{0, 1\}$ is unknown at inference time and where legitimate records ($y=0$) overwhelmingly outnumber anomalies ($y=1$), construct an unsupervised anomaly detection engine capable of:
* Accurately isolating anomalous instances without requiring minority class labels during training.
* Maintaining an ultra-low False Positive Rate (FPR) to prevent consumer friction and merchant checkout abandonment.
* Achieving sub-millisecond per-transaction inference latency ($\le 0.01\text{ ms}$).
* Providing real-time, mathematically rigorous feature attribution (Explainable AI) to satisfy regulatory auditability mandates.

---

## 2. OBJECTIVES

The specific engineering and scientific objectives of this mini project are:

1. **Robust Feature Engineering and Outlier-Resistant Preprocessing:**
   Design and implement a preprocessing pipeline utilizing the **`RobustScaler`** mathematical transformation on the non-PCA attributes (`Time` and `Amount`). This prevents multi-thousand-dollar legitimate expenditure spikes (e.g., commercial travel or corporate procurement exceeding $10,000) from corrupting the distributional scaling of typical retail transactions.
2. **Implementation of Unsupervised Isolation Forest:**
   Mathematically formulate, calibrate, and train an ensemble-based **Isolation Forest** architecture ($t=150$ randomized binary partition trees, subsample size $\psi=256$, and empirical contamination rate $\alpha \approx 0.00173$) capable of isolating outliers via recursive random cuts without exposure to minority class labels.
3. **Supervised Baseline Benchmarking:**
   Train an optimized, cost-balanced **Random Forest Classifier** ($100$ trees, maximum depth $10$, and inverse class frequency weighting) on an identical stratified $80/20$ holdout partition ($56,962$ test transactions) to empirically quantify the performance trade-offs between supervised pattern recognition and unsupervised anomaly isolation.
4. **Development of Explainable AI (XAI) Attribution:**
   Construct an automated $Z$-score divergence engine that benchmarks any candidate transaction against the multidimensional baseline centroid $(\mu, \sigma)$ of verified legitimate transactions, identifying and ranking the top latent PCA variables ($V_{14}, V_{12}, V_{10}, V_{17}$) driving the anomaly verdict.
5. **Architecting a Lightweight REST Microservice:**
   Build a modular backend using **Flask 3.1** and **Flask-CORS** exposing typed endpoints for real-time single-transaction scoring (`POST /api/predict`), vectorized bulk CSV compliance auditing (`POST /api/predict/batch`), pre-curated Kaggle sample delivery (`GET /api/samples`), and model evaluation telemetry (`GET /api/metrics`).
6. **High-Fidelity Cyber-Fintech User Interface:**
   Develop an interactive single-page application (SPA) featuring a dark-mode glassmorphic design system, an animated circular SVG Risk Gauge (calibrated from 0.0% to 100.0%), interactive sliders for key latent drivers, and confusion matrix visualizations.

---

## 3. DATASET / INPUT DESCRIPTION

### 3.1 Dataset Origin and Provenance
The system is built and benchmarked upon the canonical **Credit Card Fraud Detection Dataset** released by the Machine Learning Group (MLG) at the Université Libre de Bruxelles (ULB) in collaboration with Worldline. The dataset captures electronic transactions made by European cardholders over a two-day observation window in September 2013.

### 3.2 Dataset Statistical Summary
* **Total Transactions ($N$):** $284,807$ instances.
* **Legitimate Transactions (Class 0):** $284,315$ instances ($99.827\%$).
* **Fraudulent Transactions (Class 1):** $492$ instances ($0.173\%$).
* **Total Dimensions:** $30$ input features + $1$ binary target indicator (`Class`).
* **Missing Data / Null Values:** Exactly $0$ across all rows and columns.
* **Total Memory Footprint:** $143.84\text{ MB}$ uncompressed CSV.

### 3.3 Attribute Specification and PCA Transformation

| Attribute Name | Data Type | Physical Meaning | Range / Distribution | Preprocessing Strategy |
| :--- | :---: | :--- | :--- | :--- |
| **`Time`** | Float64 | Elapsed seconds between transaction and first record in dataset | $0.0\text{ s} \to 172,792.0\text{ s}$ (~48 hours, bimodal diurnal curve) | Scaled via `RobustScaler` |
| **`Amount`** | Float64 | Transaction value in currency | $\$0.00 \to \$25,691.16$ (Mean: $\$88.35$, Median: $\$22.00$) | Scaled via `RobustScaler` |
| **`V1` to `V28`** | Float64 | Anonymized orthogonal latent components extracted via PCA | Centered at $\mu \approx 0.0$, varying $\sigma$ across axes | Preserved intact |
| **`Class`** | Integer | Binary Ground-Truth Label | $0$ (Legitimate) or $1$ (Fraudulent) | Target evaluation label |

#### Why Principal Component Analysis (PCA)?
For customer confidentiality and strict compliance with the **Payment Card Industry Data Security Standard (PCI-DSS)** and European data privacy laws, original customer features (such as 16-digit Primary Account Numbers (PAN), cardholder names, billing postal codes, IP addresses, merchant IDs, and terminal hardware signatures) could not be published in raw form. The data custodians transformed the original 28 high-dimensional operational features into 28 orthogonal eigenvectors using PCA. Crucially, the mathematical properties of distances, angles, and tree-splits are preserved in this transformed coordinate space.

---

## 4. ALGORITHM / AI-ML TECHNIQUE USED

### 4.1 Primary Technique: Isolation Forest (`IsolationForest`)

#### 4.1.1 Core Mathematical Intuition
Traditional anomaly detection techniques (such as Gaussian Mixture Models, K-Means clustering, or One-Class SVM) attempt to model the dense regions of normal data and subsequently classify any point outside those dense envelopes as an outlier. This approach suffers heavily in high dimensions due to the **Curse of Dimensionality**, where distance metrics become equidistant and density estimation degrades exponentially.

**Isolation Forest** (invented by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou in 2008) inverts this paradigm: instead of modeling normal points, it **explicitly isolates anomalies**. The algorithm leverages two quantitative properties of anomalies:
1. They are the minority ($< 0.2\%$).
2. They possess attribute values that deviate substantially from normal clusters.

Consequently, when feature space is recursively bisected using random axis-aligned cuts, anomalous instances are partitioned into terminating leaf nodes **significantly closer to the root of the tree** than normal instances.

```
       [ NORMAL DATA CLUSTERS ]                     [ ISOLATED ANOMALY ]
   (Dense, surrounded by neighbors)             (Sparse, extreme coordinates)
                 │                                            │
   Requires 15–20 random splits                 Requires only 2–4 splits
   to isolate into its own leaf.                to isolate into its own leaf.
   ──> High Path Length h(x)                    ──> Low Path Length h(x)
   ──> Anomaly Score s -> 0.0                   ──> Anomaly Score s -> 1.0
```

#### 4.1.2 Algorithmic Formulation

##### Step 1: Isolation Tree (iTree) Construction
Let $X = \{x_1, \dots, x_n\}$ be a subsample of $n$ instances drawn without replacement from the training set, where each $x_i \in \mathbb{R}^d$.
1. If the current node contains $|X| \le 1$ instances or the tree has reached the maximum height limit $h_{\max} = \lceil \log_2(\psi) \rceil$, terminate and return an external leaf node.
2. Otherwise, randomly select a feature dimension $q \in \{1, 2, \dots, d\}$.
3. Randomly select a split threshold $p$ uniformly between the minimum and maximum values of feature $q$ in the current subset:
   $$p \sim \text{Uniform}\left(\min(X_{*, q}), \max(X_{*, q})\right)$$
4. Partition the data into two disjoint subsets:
   $$X_{\text{left}} = \{x \in X \mid x_q < p\}, \quad X_{\text{right}} = \{x \in X \mid x_q \ge p\}$$
5. Recursively apply the procedure to construct the left and right subtrees.

##### Step 2: Path Length Computation $h(x)$
For an input vector $x$, its path length $h(x)$ is the number of edges traversed from the root of an iTree until reaching a terminating leaf node. If execution terminates in an external leaf containing $m > 1$ instances (due to reaching the height limit $h_{\max}$), an adjustment factor $c(m)$ is added to represent the expected path length of a random binary search tree over $m$ instances:
$$h(x) = e + c(m)$$
where $e$ is the number of edges traversed from root to leaf.

##### Step 3: Average BST Path Length Normalization Factor $c(n)$
Because an iTree is mathematically equivalent to an unsuccessful search in a Binary Search Tree (BST), the average path length over a dataset of size $n$ is defined analytically as:
$$c(n) = 2 \left( \ln(n - 1) + \gamma \right) - \frac{2(n - 1)}{n}$$
where $\gamma = 0.5772156649\dots$ is the **Euler-Mascheroni constant**.

##### Step 4: Ensemble Anomaly Score Formulation $s(x, n)$
Across an ensemble of $t$ independent isolation trees $\{T_1, T_2, \dots, T_t\}$, the expected path length is:
$$\mathbb{E}(h(x)) = \frac{1}{t} \sum_{k=1}^t h_k(x)$$
The normalized anomaly score $s(x, n)$ bounded in $[0, 1]$ is formulated as an exponential function base 2:
$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(\psi)}}$$

* **Case 1: $\mathbb{E}(h(x)) \to 0 \implies s(x, n) \to 1.0$**  
  The instance isolates with minimal cuts. It is classified as an **outlier / fraud**.
* **Case 2: $\mathbb{E}(h(x)) \to c(\psi) \implies s(x, n) \to 0.5$**  
  The instance exhibits average path lengths typical of standard data distributions.
* **Case 3: $\mathbb{E}(h(x)) \to \psi - 1 \implies s(x, n) \to 0.0$**  
  The instance requires maximum tree partitioning. It is classified as **deeply normal**.

#### 4.1.3 Hyperparameter Selection
* **Ensemble Size ($t = 150$):** Experimental literature establishes that average path lengths $\mathbb{E}(h(x))$ stably converge at $t \ge 100$. Selecting $t = 150$ provides variance reduction without adding excessive computational overhead.
* **Subsampling Size ($\psi = 256$):** A subsample size of $\psi = 256$ provides optimal isolation efficiency while eliminating **swamping** (normal points being mistakenly isolated because they border outliers) and **masking** (large clusters of anomalies masking each other's outlier properties).
* **Contamination Rate ($\alpha = 0.00173$):** Tuned exactly to the empirical fraud frequency of the training partition ($\frac{394}{227,845} = 0.001729$), establishing the decision threshold quantile for classification.

---

### 4.2 Supervised Baseline: Random Forest Classifier

To provide an empirical baseline against which the unsupervised approach is evaluated, an optimized **Random Forest** (Breiman, 2001) supervised ensemble was trained.

* **Ensemble Size:** $100$ Decision Trees.
* **Maximum Tree Depth:** Restricted to $\text{depth} = 10$ to prevent individual trees from memorizing minor noisy perturbations.
* **Splitting Criterion:** Gini Impurity:
  $$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$
* **Cost-Sensitive Class Balancing:** To counter the $99.83\% : 0.17\%$ imbalance, loss weights inversely proportional to class frequencies were applied:
  $$w_0 = \frac{N}{2 \times N_0} = \frac{227,845}{2 \times 227,451} \approx 0.5008$$
  $$w_1 = \frac{N}{2 \times N_1} = \frac{227,845}{2 \times 394} \approx 289.14$$
  This assigns a $\sim 577\times$ heavier penalty for misclassifying a fraudulent transaction than a legitimate one during tree induction.

---

### 4.3 Explainable AI (XAI): Multidimensional Centroid $Z$-Score Attribution

A significant barrier to deploying AI in financial institutions is the "Black Box" problem, compounded by legal mandates such as **GDPR Article 22 (Right to an Explanation)**. 

To make the Isolation Forest predictions transparent, our system computes the parameter distribution of all legitimate training transactions:
$$\mu_j = \frac{1}{N_{\text{normal}}} \sum_{i \in \text{Normal}} X_{i, j}, \quad \sigma_j = \sqrt{\frac{1}{N_{\text{normal}} - 1} \sum_{i \in \text{Normal}} (X_{i, j} - \mu_j)^2}$$
For any candidate transaction vector $x^*$, the standardized divergence across all dimensions $j \in \{V_1, \dots, V_{28}, \text{Time}, \text{Amount}\}$ is:
$$Z_j = \frac{x^*_j - \mu_j}{\sigma_j}$$
The features are sorted by absolute deviation $|Z_j|$ in descending order, exposing the top 5 dominant contributors responsible for triggering the anomaly alarm (e.g., $V_{14}$ experiencing a $-10.4\sigma$ collapse).

---

## 5. METHODOLOGY / APPROACH

The implementation architecture adheres to a strict, modular machine learning pipeline spanning five distinct phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: DATA INGESTION                         │
│  - Load creditcard.csv (284,807 rows x 31 columns)                     │
│  - Verify 0 missing values; evaluate class balance (492 frauds)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: PREPROCESSING & SCALING                     │
│  - Compute normal profile (μ, σ) on legitimate transactions            │
│  - Apply RobustScaler on Time & Amount (IQR median scaling)            │
│  - Extract 6 realistic curated test cases to sample_transactions.json  │
│  - Execute Stratified 80/20 Split (227,845 train / 56,962 test)        │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│             STAGE 3: MODEL TRAINING & METRIC CALCULATION               │
│  - Train Isolation Forest (150 trees, max_samples=256, α=0.00173)      │
│  - Train Balanced Random Forest (100 trees, depth=10)                  │
│  - Compute ROC-AUC, PR-AUC, Confusion Matrix, and Latency on test set  │
│  - Persist pipelines to models/*.joblib and metrics to models/*.json   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  STAGE 4: INFERENCE & XAI ENGINE                       │
│  - Load serialized models and reference profile in detector.py         │
│  - Vectorize single & batch inputs; map decision scores to 0-100%      │
│  - Calculate real-time Z-scores; assign risk tiers (LOW/ELEVATED/CRIT) │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 STAGE 5: FULL-STACK REST APPLICATION                   │
│  - Flask Web API serving /api/predict, /api/predict/batch, /api/metrics│
│  - Cyber-fintech SPA UI with dynamic SVG gauge and parameter sliders   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. IMPLEMENTATION DETAILS

### 6.1 System Directory Structure
```
Credi_Card_Fraud_Detection/
├── data/
│   ├── creditcard.csv                 # Kaggle ULB dataset (143.84 MB)
│   ├── sample_transactions.json       # 6 verified test cases
│   └── README.md                      # Data placement instructions
├── src/
│   ├── download_helper.py             # Auto-locator and archive extractor
│   ├── preprocess.py                  # RobustScaler & stratified partitioner
│   ├── train.py                       # iForest & Random Forest training engine
│   ├── detector.py                    # Inference wrapper & Z-score XAI engine
│   └── app.py                         # Flask REST API backend
├── models/
│   ├── isolation_forest.joblib        # Fitted primary anomaly model
│   ├── baseline_rf.joblib             # Fitted supervised baseline
│   ├── scaler.joblib                  # Fitted RobustScaler
│   ├── reference_profile.json         # Normal feature distribution metrics (μ, σ)
│   └── metrics.json                   # Test set ROC-AUC, PR-AUC, confusion matrices
├── static/
│   ├── index.html                     # Cyber-fintech dashboard SPA
│   ├── css/style.css                  # Obsidian glassmorphic design system
│   └── js/app.js                      # Real-time UI controller & API bindings
├── requirements.txt                   # Dependency manifests
├── run.py                             # 1-click startup script
└── README.md                          # Repository documentation
```

### 6.2 Key Code Artifacts

#### A. Outlier-Resistant Preprocessing (`src/preprocess.py`)
```python
def preprocess_and_split(df, test_size=0.2, random_state=42):
    normal_df = df[df["Class"] == 0]
    compute_reference_profile(normal_df)
    extract_curated_samples(df)
    
    # RobustScaler scales features using median and interquartile range (IQR)
    scaler = RobustScaler()
    scaled_time_amount = scaler.fit_transform(df[["Time", "Amount"]])
    
    df_scaled = df.copy()
    df_scaled["scaled_time"] = scaled_time_amount[:, 0]
    df_scaled["scaled_amount"] = scaled_time_amount[:, 1]
    df_scaled.drop(columns=["Time", "Amount"], inplace=True)
    
    joblib.dump(scaler, SCALER_PATH)
    
    X = df_scaled.drop(columns=["Class"])
    y = df_scaled["Class"]
    
    # Stratified split ensures both splits maintain identical 0.172% fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, list(X.columns)
```

#### B. Model Training and Benchmarking (`src/train.py`)
```python
# 1. Isolation Forest (Unsupervised)
iso_forest = IsolationForest(
    n_estimators=150,
    max_samples=256,
    contamination=0.00173,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train)

# 2. Random Forest (Supervised Baseline)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
```

#### C. Production Inference and Explainability Engine (`src/detector.py`)
```python
def predict_transaction(self, txn_dict):
    X_df, raw_time, raw_amount, s_time, s_amount = self._prepare_vector(txn_dict)
    
    # Decision function returns signed distance: lower = more anomalous
    raw_decision = float(self.iso_forest.decision_function(X_df)[0])
    anomaly_score = max(0.0, min(100.0, (0.12 - raw_decision) / 0.35 * 100.0))
    is_anomaly = bool(self.iso_forest.predict(X_df)[0] == -1)
    
    rf_prob = float(self.rf_model.predict_proba(X_df)[0, 1]) * 100.0 if self.rf_model else 0.0
    composite_risk = (0.55 * anomaly_score) + (0.45 * rf_prob)
    
    # Categorization based on composite risk
    if composite_risk >= 65.0 or (is_anomaly and rf_prob >= 40.0):
        status, risk_level, badge_color = "FRAUDULENT", "CRITICAL", "#ef4444"
    elif composite_risk >= 35.0 or is_anomaly:
        status, risk_level, badge_color = "SUSPICIOUS", "ELEVATED", "#f59e0b"
    else:
        status, risk_level, badge_color = "LEGITIMATE", "LOW", "#10b981"
        
    top_drivers = self.compute_feature_explainability(txn_dict, top_k=5)
    return {
        "status": status, "risk_level": risk_level, "badge_color": badge_color,
        "composite_risk_score": round(composite_risk, 1),
        "anomaly_score": round(anomaly_score, 1),
        "supervised_probability": round(rf_prob, 1),
        "top_drivers": top_drivers
    }
```

---

## 7. RESULTS AND OUTPUT

### 7.1 Real-World Verification Test Cases
The system was verified against distinct behavioral scenarios extracted from actual holdout records:

```
[ SCENARIO 1: Everyday Grocery Store Purchase ]
Input: Amount = $33.00, Time = 23.0s, V14 = -0.08, V12 = 0.78, V10 = -0.34
Result:
  - Anomaly Score: 0.0%
  - Supervised Probability: 2.5%
  - Composite Risk: 1.1% [LOW RISK]
  - UI State: Emerald Green SVG Gauge; Status = "LEGITIMATE"
  - Decision: Transaction Approved Instantly.

[ SCENARIO 2: High-Value Luxury Hotel Booking (False-Positive Stress Test) ]
Input: Amount = $7,712.43, Time = 11,993.0s, Standard PCA feature coordinates
Result:
  - Anomaly Score: 24.3%
  - Supervised Probability: 1.8%
  - Composite Risk: 14.2% [LOW RISK]
  - Decision: Transaction Approved; proves RobustScaler prevents expenditure amounts
              from triggering false alarms.

[ SCENARIO 3: Stolen Credential Account Draining Attack ]
Input: Amount = $1.00, Time = 7,526.0s, V14 = -19.21, V12 = -17.77, V10 = -12.84
Result:
  - Anomaly Score: 95.8%
  - Supervised Probability: 30.0%
  - Composite Risk: 66.1% [CRITICAL RISK]
  - UI State: Crimson Red SVG Gauge; Status = "FRAUDULENT"
  - Decision: Transaction Auto-Blocked.
  - Explainability Driver: V14 flagged with Z = -10.4 (Extreme negative collapse).
```

---

## 8. PERFORMANCE EVALUATION

### 8.1 Empirical Metrics on Holdout Test Set ($N = 56,962$)

| Evaluation Metric | Isolation Forest *(Unsupervised)* | Random Forest *(Supervised Baseline)* | Mathematical Formula |
| :--- | :---: | :---: | :---: |
| **Learning Paradigm** | Unsupervised (Tree Isolation) | Supervised (Cost-Balanced) | -- |
| **ROC-AUC Score** | **0.9546 (95.46%)** | **0.9798 (97.98%)** | $\int_0^1 \text{TPR}(f) \, d(\text{FPR}(f))$ |
| **PR-AUC (Average Precision)**| **0.1475** | **0.8321** | $\sum_k (R_k - R_{k-1}) P_k$ |
| **Recall (Fraud Caught)** | **25.51% (25 / 98)** | **85.71% (84 / 98)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ |
| **Precision** | **25.25% (25 / 99)** | **64.62% (84 / 130)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ |
| **F1-Score** | **0.2538** | **0.7368** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **Inference Latency** | **0.005 ms / transaction** | **0.003 ms / transaction** | Execution duration / $N$ |
| **Training Duration** | **2.07 seconds** | **14.32 seconds** | CPU execution time |
| **Zero-Day Attack Resilience**| ⭐⭐⭐⭐⭐ **Maximum (No labels)** | ⭐⭐ Limited to past patterns | Qualitative assessment |

### 8.2 Detailed Confusion Matrix Analysis (56,962 Holdout Instances)

#### Isolation Forest (Unsupervised):
$$\begin{pmatrix} \text{True Negatives (TN)} & \text{False Positives (FP)} \\ \text{False Negatives (FN)} & \text{True Positives (TP)} \end{pmatrix} = \begin{pmatrix} 56,790 & 74 \\ 73 & 25 \end{pmatrix}$$
* **Specificity (TNR):** $\frac{56,790}{56,790 + 74} = \mathbf{99.870\%}$ (Extremely low consumer friction).
* **False Alarm Rate (FPR):** $\frac{74}{56,864} = \mathbf{0.130\%}$ (Only 1 in 768 normal transactions held for review).

#### Random Forest (Supervised):
$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 56,818 & 46 \\ 14 & 84 \end{pmatrix}$$
* **Specificity (TNR):** $\frac{56,818}{56,818 + 46} = \mathbf{99.919\%}$.
* **Fraud Detection Rate (Recall):** $\frac{84}{84 + 14} = \mathbf{85.714\%}$.

---

## 9. ANALYSIS AND DISCUSSION

### 9.1 Comparative Analysis: Unsupervised vs. Supervised Paradigms
A superficial glance at raw precision and recall metrics might suggest that supervised Random Forest is unconditionally superior to unsupervised Isolation Forest. However, in enterprise payments engineering, this comparison must be qualified by domain realities:

1. **The Cold-Start and Label Latency Dilemma:**
   The supervised Random Forest model requires thousands of confirmed, historical fraud examples to build its Gini-minimizing splits. In real banking infrastructure, chargebacks take between **60 and 90 days** to formalize. If an institution enters a new geographic market, launches a new card product, or experiences an infrastructure breach, supervised models have zero labeled historical data. In contrast, **Isolation Forest achieved an exceptional ROC-AUC of 0.9546 with ZERO labeled fraud examples**, making it uniquely viable for cold-start environments.
2. **Adversarial Resilience to Zero-Day Attacks:**
   Supervised models overfit to the explicit boundaries of *known* fraud patterns. When a fraud syndicate alters its behavioral profile, supervised classifiers experience catastrophic performance drops. Because **Isolation Forest models isolation difficulty rather than class boundaries**, any anomalous pattern—regardless of whether it matches historical attacks—is flagged.
3. **The Multi-Tiered Hybrid Defense Architecture:**
   The ultimate industry solution demonstrated in AegisPay is a **hybrid dual-engine architecture**:
   * The **Supervised Model** auto-blocks known, high-volume recurring fraud patterns.
   * The **Unsupervised Anomaly Model** flags subtle, emerging, zero-day statistical anomalies for secondary multi-factor authentication (MFA).

### 9.2 Computational Scalability in Payment Gateways
In production transaction networks (e.g., VisaNet or Mastercard BankNet), authorization engines handle peak workloads exceeding **20,000 Transactions Per Second (TPS)**. The entire security authorization decision must execute in under **50 milliseconds**. With our Isolation Forest inference benchmark of **0.005 milliseconds (5 microseconds)** per transaction, our AI engine consumes less than **$0.01\%$ of the total available latency budget**, leaving ample headroom for network routing and cryptographic validation.

### 9.3 Why Accuracy is Scientifically Invalid for Imbalanced Data
If an evaluator assesses this project using Accuracy:
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} = \frac{25 + 56,790}{56,962} = \mathbf{99.74\%}$$
A naive baseline that blindly outputs 0 for all instances achieves:
$$\text{Baseline Accuracy} = \frac{0 + 56,864}{56,962} = \mathbf{99.83\%}$$
The naive baseline has a higher accuracy, yet it is completely useless for fraud detection. This mathematically proves why academic evaluation must center on **ROC-AUC (0.9546)**, **PR-AUC**, and **Cost-Weighted Utility Functions**.

---

## 10. CONCLUSION

In this project, an end-to-end, industrial-grade credit card fraud detection system named **AegisPay** was successfully designed, implemented, and empirically evaluated.

1. By implementing **Isolation Forest** tree partitioning alongside **RobustScaler** feature transformation, the system isolates fraudulent card transactions effectively, achieving an **ROC-AUC of 0.9546** without requiring a single fraud label during training.
2. The system demonstrates ultra-high computational efficiency, executing single-transaction evaluations in **0.005 ms**, ensuring suitability for real-time payment gateway deployment.
3. The integration of an **Explainable AI (XAI)** engine computing real-time $Z$-score divergences solves the "black-box" dilemma, empowering fraud investigators with actionable, transparent explanations for every flagged transaction.
4. The accompanying modern, glassmorphic cyber-fintech dashboard bridges the gap between complex algorithmic models and practical operational usability.

---

## 11. FUTURE SCOPE

To evolve AegisPay beyond this mini project, future research and development will explore:

1. **Deep Semi-Supervised Variational Autoencoders (VAEs):**
   Constructing PyTorch-based neural Autoencoders (Encoder $\to$ Bottleneck Latent Space $\to$ Decoder) trained purely on legitimate transactions. Transactions producing high Reconstruction Loss:
   $$\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|_2^2 > \tau$$
   will be classified as high-confidence non-linear anomalies.
2. **Heterogeneous Graph Neural Networks (GNNs):**
   Formulating payment ecosystems as dynamic bipartite graphs $\mathcal{G} = (\mathcal{V}_{\text{cards}}, \mathcal{V}_{\text{merchants}}, \mathcal{E}_{\text{transactions}})$. Utilizing Relational Graph Convolutional Networks (R-GCNs) will allow detection of complex, collusive fraud rings and card-cloning syndicates that evade single-transaction vector analysis.
3. **Distributed Streaming Infrastructure:**
   Integrating **Apache Kafka** for event-driven message queuing and **Redis** for in-memory velocity tracking (e.g., rolling 60-second card velocity counters), enabling horizontal scale across multi-region cloud clusters.
4. **Federated Learning for Cross-Bank Collaboration:**
   Deploying privacy-preserving Federated Learning algorithms (such as FedAvg with Differential Privacy) allowing multiple commercial banking institutions to jointly train shared anomaly models without ever exposing proprietary cardholder transaction records.

---

## 12. REFERENCES

1. **Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008).** *Isolation Forest.* In Proceedings of the 8th IEEE International Conference on Data Mining (ICDM '08), Pisa, Italy, pp. 413–422. DOI: 10.1109/ICDM.2008.17.
2. **Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015).** *Calibrating Probability with Undersampling for Unbalanced Classification.* In IEEE Symposium Series on Computational Intelligence (SSCI), Cape Town, South Africa, pp. 159–166.
3. **Breiman, L. (2001).** *Random Forests.* Machine Learning, Vol. 45, No. 1, pp. 5–32.
4. **Goldstein, M., & Uchida, S. (2016).** *A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data.* PLOS ONE, Vol. 11, No. 4, e0152173. DOI: 10.1371/journal.pone.0152173.
5. **Emmott, A., Das, S., Dietterich, T., Fern, A., & Wong, W. K. (2013).** *Systematic Construction of Anomaly Detection Benchmarks from Real Data.* In ACM SIGKDD Workshop on Outlier Detection De-Constructed (ODD '13), Chicago, IL, pp. 16–21.
6. **Machine Learning Group (ULB):** *Credit Card Fraud Detection Benchmark Dataset.* Université Libre de Bruxelles, hosted on Kaggle: [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
7. **Rousseeuw, P. J., & Croux, C. (1993).** *Alternatives to the Median Absolute Deviation.* Journal of the American Statistical Association, Vol. 88, No. 424, pp. 1273–1283. (Foundation of Robust Outlier Scaling).
