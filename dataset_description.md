# FictiPay Churn Prediction: Dataset & Action Guide

This document details the FictiPay dataset structure, schema definitions, churn rules, engineered feature categories, and how the data is processed through the end-to-end machine learning pipeline.

---

## 1. Raw Dataset Schema

The raw dataset contains three primary tables located in `bkash-presents-nsucec-datathon/public/`:

### 1. KYC Metadata (`kyc.parquet`)
* **Size**: ~2,000,000 unique accounts (all types).
* **Columns**:
  * `ACCOUNT_ID` (String): Unique identifier of the wallet.
  * `ACCOUNT_TYPE` (String): Categorized as `Customer`, `Merchant`, or `Biller`.
  * `ACCOUNT_OPEN_DATE` (String/Date): Timestamp when the account was activated.
  * `GENDER` (String): Account holder gender (`MALE`, `FEMALE`, or missing/null).
  * `REGION` (String): Geographic area/administrative division in Emergingland.

### 2. Transaction Records (`transactions/trx_2024-0[1-3].parquet`)
* **Size**: ~200,000,000 total transaction records across January, February, and March 2024.
* **Columns**:
  * `TrxID` (String): Unique transaction identifier.
  * `TRX_DATETIME` (String/Timestamp): Date and time of transaction.
  * `SRC_ACCOUNT` (String): Sender account ID (always a `Customer` account in this dataset).
  * `DST_ACCOUNT` (String): Recipient account ID (can be a `Customer`, `Merchant`, or `Biller`).
  * `TRX_TYPE` (String): Category of transaction:
    * `P2P` (Peer-to-Peer transfer to another Customer)
    * `MerchantPay` (Payment to a Merchant)
    * `BillPay` (Utility or service bill payment to a Biller)
    * `CashIn` (Adding cash to wallet via an agent/bank)
    * `CashOut` (Withdrawing cash from wallet via an agent)
  * `TRX_AMT` (Double): Transaction amount in TK.

### 3. Day-End Balances (`dayend_balance/balance_2024-0[1-3].parquet`)
* **Size**: ~360,000,000 balance checkpoints (2M accounts × 180 days).
* **Columns**:
  * `ACCOUNT_ID` (String): Account identifier.
  * `DATE` (String/Date): Balance checkpoint date.
  * `AVAILABLE_BALANCE` (Double): Available balance in TK at the end of that day.

---

## 2. Customer-Centric Target & Churn Definition

### Churn Target
* **Target Audience**: Only **Customer** accounts (`ACCOUNT_TYPE = 'Customer'`) are scored in the training labels (`train_labels.csv`) and test IDs (`test.csv`). Merchants and Billers act strictly as transaction endpoints.
* **Observation Window**: `2024-01-01` to `2024-03-31` (90 days). All model input features must be computed strictly from this window.
* **Prediction Window**: `2024-04-01` to `2024-04-30` (30 days).
* **Ground Truth definition**:
  * **`CHURN = 1`** (Churned): The customer executed **zero** transactions (inbound or outbound) during the 30-day prediction window (April 2024).
  * **`CHURN = 0`** (Active): The customer executed **at least one** transaction (inbound or outbound) during the prediction window.

### Class Imbalance
* **Training Set Size**: 595,000 accounts.
* **Active (`CHURN = 0`)**: 519,565 accounts (87.3%).
* **Churned (`CHURN = 1`)**: 75,435 accounts (12.7%).
* **Imbalance Ratio**: ~7:1 majority class to minority class.

---

## 3. Engineered Feature Catalog

The features engineered in `features.py` (totaling ~130 features) are grouped into 8 functional domains:

### 1. Demographic & Tenure Features (KYC)
* **tenure_days**: Days from `ACCOUNT_OPEN_DATE` to `2024-03-31`. Shorter tenure indicates a new account with unestablished usage habits.
* **gender_*, region_***: One-hot encoded columns indicating gender and region.

### 2. General Transaction Volume (Jan + Feb + March)
* **out_count_[month/total]** / **in_count_[month/total]**: Counts of outbound and inbound transactions per month.
* **out_sum_[month/total]** / **in_sum_[month/total]**: Total transaction amounts in TK.
* **out_avg_[month]** / **out_std_[month]** / **out_max_[month]**: Mean, standard deviation, and maximum outbound transaction size per month.

### 3. Directional & Type Recency
* **days_since_last_trx**: Days from the customer's last transaction (any direction) to the reference date `2024-03-31`.
* **days_since_last_outbound** / **days_since_last_inbound**: Recency of outbound/inbound operations.
* **days_since_last_[P2P / MerchantPay / BillPay / CashOut]**: Recency of specific outbound transaction types.
* **days_since_last_CashIn** / **days_since_received_P2P**: Recency of specific inbound transaction types.

### 4. Micro-Windows & Velocity
* **out_count_last_7d** / **out_sum_last_7d**: Activity count and spend amount in the final week of March (March 25-31).
* **out_count_last_14d** / **out_sum_last_14d**: Activity count and spend amount in the final two weeks of March (March 18-31).
* **in_count_last_7d** / **in_sum_last_7d** / **in_count_last_14d** / **in_sum_last_14d**: Inbound micro-window stats.
* **out_count_velocity_7d** / **out_count_velocity_14d**: Ratio of micro-window counts to the overall monthly March count. A ratio near 0 indicates recent dormancy.

### 5. Temporal Decay Trends
* **trx_count_decay_jan_feb** / **trx_count_decay_feb_march** / **trx_count_decay_jan_march**: Rate of change in outbound transactions between months:
  $$\text{decay}_{A\_B} = \frac{\text{count}_B - \text{count}_A}{\text{count}_A + 1}$$
* **march_activity_share**: March transaction count divided by total transaction count. A low share reflects fading interaction.

### 6. Service Diversity & Type Ratios
* **count_[TrxType]_total**: Grand total count of each transaction type.
* **trx_type_diversity**: Number of unique transaction types used (integer 0–5). High diversity signals high customer integration.
* **merchant_pay_ratio** / **bill_pay_ratio** / **p2p_ratio** / **cashout_ratio**: Proportion of outbound transaction counts represented by specific types.
* **flag_zero_bill_pay** / **flag_zero_merchant_pay** / **flag_zero_march_trx** / **flag_zero_trx_last_7d_march**: Indicator flags representing zero usage in key categories.

### 7. Net Flow
* **net_flow_march**: March total inbound volume minus outbound volume:
  $$\text{net\_flow} = \text{in\_sum\_march} - \text{out\_sum\_march}$$
  A highly negative net flow indicates a customer drawing down their wallet balance.

### 8. Daily Balance Trends & Micro-Windows
* **mean_bal_[month]** / **std_bal_[month]** / **min_bal_[month]** / **max_bal_[month]**: Basic balance descriptors.
* **final_balance_march**: Available balance on the final observation day (March 31).
* **balance_drop_march**: Balance difference between March 31 and March 1.
* **balance_trend_march**: The linear slope of daily balances over the 31 days of March.
* **zero_balance_days_march**: Total days where available balance was $<10$ TK.
* **balance_stability_march**: Coefficient of Variation (CV) of March daily balances:
  $$\text{stability} = \frac{\text{std\_bal\_march}}{\text{mean\_bal\_march} + 1\text{e-}5}$$
* **final_to_mean_balance_ratio_march**: Final day's balance divided by mean March balance.

---

## 4. Putting the Dataset into Action: Pipeline Execution Flow

The project processes, models, and scores this dataset through six key actions:

```
[Raw Files] 
    │
    ▼ (1. Big Data Load & Downscaling)
[Dask / Column Pruning / March Path Configured]
    │
    ▼ (2. Distribution Adjustments)
[log1p / Sparsity Flags / Demographic Imputation]
    │
    ▼ (3. Validation Partitioning)
[Stratified 10-Fold CV Splits]
    │
    ▼ (4. Classifier Zoo)
[LGBM / XGBoost / CatBoost]
    │
    ▼ (5. Blending & Probability Correction)
[AUC-Squared Rank Averaging + Isotonic Regression]
    │
    ▼ (6. Final Scoring & Formatting)
[Expected Loss Thresholding -> predictions.csv]
```

### 1. Memory-Safe Partition Processing
* The script processes large tables by applying column pruning (only loading the IDs, datetime, and amount columns) and running memory-safe aggregations monthly.
* March daily balances are aggregated using a chunked pivot loop (processing 100,000 account IDs per iteration) to calculate slopes and trends without overloading RAM.

### 2. Distribution Preprocessing
* **Right-Skewed monetary amounts** (sums, max values, balances) are stabilized by applying a $log(x + 1)$ scaling.
* **Zero-inflation** is handled by generating explicit binary flags for variables containing substantial zero counts (such as recent transactions and bill payments).

### 3. Stratified Partitioning & Training Zoo
* The training data is split into 10 folds using a Stratified K-Fold partition. This maintains the 12.7% churn rate across all validation sets.
* Models are trained sequentially. Hardware settings dynamically check PyTorch CUDA to run boosting models on GPU when available, falling back to CPU with thread optimization (`n_jobs=-1`).

### 4. Kaggle-Optimal Blending
* Predictions from the model zoo are combined using **Rank-Average Blending**, converting probabilities to percentiles (ranks 0–1) weighted by the squared cross-validation AUC of the models:
  $$W_i = \frac{\text{AUC}_i^2}{\sum \text{AUC}_j^2}$$
* This rank blend is mapped back to calibrated probabilities by fitting a strictly monotonic `LogisticRegression` curve (Platt Scaling) on the validation ranks to prevent granularity loss.

### 5. Cost-Sensitive Optimization
* The final probabilities are evaluated against a business loss function where missing a churner costs $5\times$ more than a false alarm:
  $$\text{Cost} = 5 \times \text{False Negative} + 1 \times \text{False Positive}$$
* The system sweeps decision thresholds from 0.1 to 0.9 to locate the cut-off point that minimizes total business cost.
* Final predictions are verified and exported in `predictions.csv`.
