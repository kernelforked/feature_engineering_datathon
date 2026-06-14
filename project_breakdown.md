# FictiPay Churn Prediction: Project Codebase Breakdown

This document provides a detailed breakdown of all files and scripts in the FictiPay churn prediction project. It explains the purpose, key functions, dependencies, and execution flow of each component.

---

## 1. Directory Structure

```
/home/aspen/ProjectCollections/keggle/Datathon/
├── bkash-presents-nsucec-datathon/       # Raw dataset containing 'public/'
│   └── public/
│       ├── dayend_balance/               # Daily wallet balance parquets (Jan, Feb, March)
│       ├── transactions/                 # Monthly transaction parquets (Jan, Feb, March)
│       ├── kyc.parquet                   # Customer demographic metadata
│       ├── train_labels.csv              # Customer IDs and churn labels
│       ├── test.csv                      # Test customer IDs to predict
│       └── sample_submission.csv         # Submission template format
├── processed_data/                       # Intermediate engineered features [Parquet]
├── models/                               # Serialized model pickles
├── predictions/                          # Out-Of-Fold and Test predictions [Parquet/CSV]
├── plots/                                # Performance and interpretability plots
├── run_pipeline.py                       # End-to-end pipeline orchestrator
├── features.py                           # Feature engineering script
├── train.py                              # Stratified K-Fold classifier training
├── ensemble.py                           # Blending, ensembling, and calibration
├── explain.py                            # SHAP values and data leakage audit
├── segment.py                            # PyTorch Autoencoder + K-Means clustering
├── fix_submission.py                     # Submission formatting patch
├── handover.md                           # Takeover handover documentation
├── Checklist.txt                         # IBM Telco checklist mapping guideline
└── report.tex                            # LaTeX project documentation report
```

---

## 2. Detailed File Explanations

### 1. [run_pipeline.py](file:///home/aspen/ProjectCollections/keggle/Datathon/run_pipeline.py)
* **Purpose**: Serves as the central orchestrator that executes the pipeline steps in a modular and reproducible manner.
* **Flow**:
  1. Calls `features.py` to run the feature engineering stage.
  2. Calls `train.py` to fit the models in the Model Zoo.
  3. Calls `ensemble.py` to perform rank-average blending, calibration, and write out predictions.
* **Dependencies**: `subprocess`, `sys`, `time`

### 2. [features.py](file:///home/aspen/ProjectCollections/keggle/Datathon/features.py)
* **Purpose**: Engages in big data feature engineering, generating over 130 features from raw KYC, monthly transactions, and daily balances.
* **Key Functions**:
  * `load_kyc(target_ids)`: Loads metadata, calculates tenure relative to 2024-03-31, imputes missing values, and one-hot encodes `GENDER` and `REGION`.
  * `compute_recency(target_ids)`: Performs a lightweight column-pruned datetime pass over the transactional files to extract `days_since_last_trx`.
  * `compute_advanced_march_features(target_ids)`: Extracts last 7-day and 14-day spending velocities and directional recency for March.
  * `aggregate_transactions_one_month(...)`: Accumulates sums, counts, means, standard deviations, and transaction types for a specific month.
  * `aggregate_transactions(target_ids)`: Sequences the monthly aggregations and constructs cross-month features (decay metrics and transaction ratios).
  * `aggregate_balances(target_ids)`: Aggregates balances month-by-month and runs a chunked pivot (chunk size: 100K) over March daily balances to extract balance slope/trends and zero-balance days without running out of RAM.
  * `build_features(...)`: Connects all features, adds sparsity flags, executes $log(x+1)$ transforms for skewed columns, splits the accounts into training and test sets, and outputs parquet files.
* **Audit and Path Resolution**:
  > [!WARNING]
  > The original code contained a hardcoded base path:
  > `base_path = "/home/ahnaf-zakaria/Desktop/Datathon/bkash-presents-nsucec-datathon/public"`
  > Because this folder did not exist on the target system, runs fell back to skipping March transaction features under the assumption they were "missing".
  > By updating `base_path` to `/home/aspen/ProjectCollections/keggle/Datathon/bkash-presents-nsucec-datathon/public`, the pipeline successfully discovers and integrates `trx_2024-03.parquet` and `balance_2024-03.parquet`, adding valuable transactional recency and velocity features.

### 3. [train.py](file:///home/aspen/ProjectCollections/keggle/Datathon/train.py)
* **Purpose**: Establishes a Model Zoo and trains baseline classifiers using 10-Fold Stratified Cross-Validation to validate predictions.
* **Classifiers Profile**:
  1. **LightGBM**: Tuned GBDT with deeper configurations (`num_leaves=63`, `max_depth=8`) and balanced class weighting.
  2. **XGBoost**: Histogram-based tree method with early stopping to prevent overfitting.
  3. **CatBoost**: GPU-enabled GBDT classifier (falls back to CPU if not installed).
  4. **Random Forest**: 150 trees of depth 12 acting as a robust bagging baseline.
  5. **Logistic Regression** and **MLP Classifier (Neural Network)**: Multilayer perceptron (128-64 hidden nodes) trained on scaled and imputed features.
* **Key Functions**:
  * `get_xgb_device_params()` / `get_lgb_device_params()` / `get_catboost_device_params()`: Queries device properties (PyTorch CUDA or dummy trainings) to automatically toggle GPU acceleration or CPU thread counts (`n_jobs=-1`).
  * `train_***(...)`: CV loops capturing Out-Of-Fold (OOF) predictions and test predictions.
  * `plot_curves(y_true, oof_dict)`: Saves validation ROC curves to `plots/roc_curves.png`.
* **Output**: Writes `oof_predictions.parquet`, `test_predictions.parquet`, and model pickle binaries in `/models`.

### 4. [ensemble.py](file:///home/aspen/ProjectCollections/keggle/Datathon/ensemble.py)
* **Purpose**: Blends model predictions to maximize the ROC-AUC score and calibrates outputs for exact business risk profiling.
* **Pipeline logic**:
  1. **Stacking Meta-Classifier**: Trains a Logistic Regression meta-model over stacked base OOF predictions using 5-fold CV.
  2. **Weighted Average**: A baseline soft-voting ensemble weighted by the base model validation AUCs.
  3. **Rank-Average Blending**: Map individual models' predictions to relative percentiles (ranks) before averaging. This is mathematically optimal for ROC-AUC as it preserves ordinal relationships and removes model-specific calibration differences.
  4. **Probability Calibration**: Fits an `IsotonicRegression` wrapper on the rank-average blended values to output valid class probabilities.
  5. **Cost-Sensitive Decision Threshold**: Computes an optimal decision threshold targeting the competition cost matrix ($5 \times FN + 1 \times FP$).
* **Output**: Generates `predictions.csv` and `test_probabilities.csv`.

### 5. [explain.py](file:///home/aspen/ProjectCollections/keggle/Datathon/explain.py)
* **Purpose**: Provides model interpretability, feature importance rankings, and standalone feature leakage audits.
* **Key Operations**:
  * Samples 5,000 instances to run SHAP `TreeExplainer` on the LightGBM classifier.
  * Exports `shap_beeswarm.png` showing directional impacts (e.g., how low activity triggers churn risk).
  * Exports `shap_bar.png` ranking features by mean absolute SHAP values.
  * Generates four subplots in `shap_dependence.png` demonstrating non-linear interactions.
  * Audits top features by evaluating standalone ROC-AUC. A standalone AUC $> 0.95$ flags potential target leakage (e.g., features leaking post-reference events).

### 6. [segment.py](file:///home/aspen/ProjectCollections/keggle/Datathon/segment.py)
* **Purpose**: Translates predictions into actionable customer groups.
* **Architecture**:
  1. **PyTorch Autoencoder**: Learns a 16-dimensional latent space representation of behavioral variables.
  2. **K-Means Clustering**: Groups the latent representations into $K$ clusters ($K \in \{3,4,5\}$ evaluated via Silhouette Score on a 5K sub-sample).
  3. **Segment Profiles**: Aggregates demographic and activity metrics per cluster and details personalized retention interventions (e.g. loyalty cashback vs. re-engagement campaigns).
* **Output**: Generates `plots/segmentation.png` and `plots/segment_profiles.csv`.

### 7. [fix_submission.py](file:///home/aspen/ProjectCollections/keggle/Datathon/fix_submission.py)
* **Purpose**: A final-step formatting helper. Ensures that the ensembled test probability output has correct columns (`ACCOUNT_ID`, `CHURN_PROB`) and matches the target submission format.

---

## 3. Pipeline Interaction Schema

```mermaid
flowchart TD
    subgraph Raw Data Ingestion
        A[kyc.parquet]
        B[transactions/trx_2024-*.parquet]
        C[dayend_balance/balance_2024-*.parquet]
    end

    subgraph Feature Engineering (features.py)
        D[load_kyc]
        E[compute_recency]
        F[compute_advanced_march_features]
        G[aggregate_transactions]
        H[aggregate_balances]
        I[build_features]
        
        A --> D
        B --> E
        B --> F
        B --> G
        C --> H
        
        D & E & F & G & H --> I
    end

    subgraph Data Flow
        J[(processed_data/train_features.parquet)]
        K[(processed_data/test_features.parquet)]
        I --> J
        I --> K
    end

    subgraph Training Zoo (train.py)
        L[10-Fold Stratified CV]
        M[LightGBM]
        N[XGBoost]
        O[CatBoost]
        P[Random Forest]
        Q[Linear & MLP Models]
        
        J & K --> L
        L --> M & N & O & P & Q
    end

    subgraph Intermediate Outputs
        R[(predictions/oof_predictions.parquet)]
        S[(predictions/test_predictions.parquet)]
        T[plots/roc_curves.png]
        
        M & N & O & P & Q --> R
        M & N & O & P & Q --> S
        M & N & O & P & Q --> T
    end

    subgraph Ensembling (ensemble.py)
        U[Rank-Average Blending]
        V[Isotonic Calibration]
        W[Cost-Sensitive Thresholding]
        X[predictions.csv]
        
        R & S --> U
        U --> V
        V --> W
        W --> X
    end

    subgraph Analytics & Interpretation
        R --> Y[explain.py: SHAP Explanations]
        R --> Z[segment.py: Latent Clustering]
    end
```
