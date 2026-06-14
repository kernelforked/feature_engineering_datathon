# features.md — FictiPay Churn Prediction

## 1. Project Overview

### Business Problem
FictiPay is a mobile wallet application facing customer churn. Retaining existing users is significantly more cost-effective than acquiring new ones. This document defines and details the feature store architecture designed to predict customer churn, enabling the marketing and product teams to target high-risk users with proactive retention campaigns.

### Dataset Summary
The pipeline processes raw, large-scale financial and customer data from three distinct corporate data sources:
1. **KYC Metadata (`kyc.parquet`)**: Demographic data including region, gender, and account creation timestamps for all registered accounts.
2. **Transaction Logs (`trx_2024-*.parquet`)**: Raw transaction records spanning January and February 2024 (~200 million rows). Due to a raw data limitation in the public release, March transaction data is completely missing.
3. **Daily Balance Logs (`dayend_balance_*.parquet`)**: Daily wallet ending balances spanning January, February, and March 2024 (~360 million rows). March balance logs are fully intact, serving as a critical activity signal for the final observation month.

### Feature Store Dimensions & Scale
* **Total Engineered Features**: **137 features** (producing a final output matrix of 139 columns, including `ACCOUNT_ID` and the `CHURN` target).
* **Training Set Dimensions**: 595,000 customers × 137 features.
* **Test Set Dimensions**: 255,000 customers × 137 features.
* **Event Log Reduction**: Aggregates ~200 million transactional logs and ~360 million daily balances into 137 behavioral markers per customer.

---
To prevent data leakage and ensure realistic modeling, the time logic is structured as follows:
* **Observation Window**: January 1, 2024 – March 31, 2024 (90 days).
* **Prediction Window**: April 1, 2024 – April 30, 2024 (30 days).
* **Training Target (CHURN)**: 
  $$\text{CHURN} = \begin{cases} 1 & \text{if a customer executes zero transactions in April 2024} \\ 0 & \text{otherwise} \end{cases}$$
* **Scope**: Only accounts classified as `CUSTOMER` in the KYC records are included in training. Merchant and system agent accounts are filtered out.

---

## 2. Feature Engineering Philosophy

### Limits of Raw Transaction Logs
Raw transaction logs cannot be fed directly into machine learning models. They represent transactional time-series of variable length (e.g., one customer might transact 1,000 times, another only twice). A static feature vector is required.

### The Necessity of Aggregation
Aggregation acts as a lossy compression mechanism. It translates thousands of transactional rows into standardized metrics (counts, sums, averages, variances) representing a customer's spending capacity, transaction frequency, and savings patterns.

### The Role of Time-Windowed Behavior
Human behavior is dynamic. Transactional behavior exhibits temporal decay: a customer's transactions in January are less indicative of their churn risk in April than their transactions in March. Therefore, features must be engineered across multiple temporal windows (e.g., last 7 days, last 14 days, month-by-month) to capture the velocity and acceleration of user disengagement.

### Modeling Churn via Behavioral Signals
In digital wallets, customers rarely close their accounts explicitly. Instead, they "fade away" (silent churn). This disengagement manifests as:
* Wallet draining (steadily decreasing balances).
* Increasing dormancy (growing intervals between transactions).
* Product disuse (switching from high-stickiness merchant payments to cash-outs).

By capturing these behavioral signals, the feature store exposes the early warning signs of silent churn.

---

## 3. Feature Design Principles

### 3.1 Time-Awareness Principle
To guarantee validation integrity, all features are strictly computed using historical data recorded prior to **2024-03-31 23:59:59**. Any data point from April 1, 2024, or later is strictly partitioned off during feature extraction.

### 3.2 Customer-Centric Aggregation
The primary key of the feature store is `ACCOUNT_ID`. All raw transaction events and daily balance records are mapped, aggregated, and joined at the `ACCOUNT_ID` level, ensuring a shape of `(N_customers, N_features)` for training.

### 3.3 Streaming-First Design
Given the size of the raw datasets (~200M transactions, ~360M daily balances), loading the datasets into RAM is impossible on standard infrastructure. The feature extraction logic is designed using chunked streaming. Transaction logs are loaded in blocks, aggregated incrementally, and combined using memory-efficient out-of-core operations.

### 3.4 No-Leakage Principle
Features must never incorporate future knowledge. For example, computing a user's "average transaction size" across both the observation and prediction windows is a severe leak that artificially inflates validation metrics. The prediction window (April) remains completely invisible to the feature engineering pipeline.

### 3.5 Stability Principle
All engineered features must remain mathematically consistent across different data partitions. Standardizations and imputations must be calculated on the training splits and applied out-of-fold to validation/test splits to avoid cross-fold distribution shifts.

---

## 4. Feature Groups

### 4.1 Recency Features

#### Feature: `days_since_last_trx`
* **Definition**: The number of days between the user's most recent transaction (inbound or outbound) and the observation boundary (`2024-03-31`).
* **Business Meaning**: Measures the current period of wallet dormancy.
* **Why it Predicts Churn**: A long duration of inactivity indicates that the customer has likely substituted FictiPay with a competitor or cash.
* **Computation Logic**: 
  $$\text{days\_since\_last\_trx} = 2024\text{-}03\text{-}31 - \max(\text{transaction\_timestamp})$$
* **Data Source**: `trx_2024-01.parquet`, `trx_2024-02.parquet` (and March balance records for wallet activity proxies).
* **Leakage Safety Check**: Strictly bounded by the 2024-03-31 cutoff.

---

### 4.2 Frequency Features

#### Feature: `out_count_total`
* **Definition**: The total number of outbound transactions executed by the customer during January and February.
* **Business Meaning**: Represents the transactional habit strength of the user.
* **Why it Predicts Churn**: High frequency indicates a highly integrated user who relies on FictiPay for regular operations. Low frequency represents a casual user who can easily churn.
* **Computation Logic**: 
  $$\sum \mathbb{I}(\text{direction} = \text{'outbound'})$$
* **Data Source**: `trx_2024-01.parquet`, `trx_2024-02.parquet`
* **Leakage Safety Check**: Uses only January and February transactional logs.

---

### 4.3 Monetary Features

#### Feature: `out_sum_total`
* **Definition**: The total currency volume (in TK) transacted outbound by the customer in January and February.
* **Business Meaning**: Represents the customer's financial volume and overall monetary value to the wallet.
* **Why it Predicts Churn**: Customers moving large volumes of money have high trust in the platform and are deeply embedded. Small-value accounts are more volatile and prone to churn.
* **Computation Logic**: 
  $$\sum \text{amount} \quad \text{where} \quad \text{direction} = \text{'outbound'}$$
* **Data Source**: `trx_2024-01.parquet`, `trx_2024-02.parquet`
* **Leakage Safety Check**: Restricted to the pre-March observation logs.

---

### 4.4 Balance Features

#### Feature: `final_balance_march`
* **Definition**: The customer's wallet ending balance on March 31, 2024.
* **Business Meaning**: The amount of liquid capital currently stored in the wallet.
* **Why it Predicts Churn**: Users intending to churn typically drain their wallets. A final balance of near-zero is a strong indicator of imminent abandonment.
* **Computation Logic**: 
  $$\text{balance} \quad \text{where} \quad \text{date} = 2024\text{-}03\text{-}31$$
* **Data Source**: `dayend_balance_march.parquet`
* **Leakage Safety Check**: Evaluated precisely at the final boundary of the observation window.

---

### 4.5 Transaction Type Features

#### Feature: `cashout_ratio`
* **Definition**: The ratio of Cash-Out transaction volume to the total outbound volume.
* **Business Meaning**: Measures the extent to which a user treats the digital wallet as a temporary bridge to paper cash rather than a digital payment ecosystem.
* **Why it Predicts Churn**: High cash-out ratios indicate the user immediately extracts deposited funds. Users with high P2P or Merchant Pay ratios are actively using the ecosystem, which indicates higher retention.
* **Computation Logic**: 
  $$\frac{\sum \text{amount} \quad \text{where} \quad \text{type} = \text{'CashOut'}}{\sum \text{amount} \quad \text{where} \quad \text{direction} = \text{'outbound'}}$$
* **Data Source**: `trx_2024-01.parquet`, `trx_2024-02.parquet`
* **Leakage Safety Check**: Computed using historical data prior to March 31.

---

### 4.6 Temporal Decay Features

#### Feature: `trx_count_decay_jan_feb`
* **Definition**: The ratio of February transaction count to January transaction count.
* **Business Meaning**: Captures month-over-month usage velocity (growth vs. decay).
* **Why it Predicts Churn**: A decay ratio $< 1.0$ indicates that the customer is transacting less frequently month-over-month. This downward trajectory is a classic early warning indicator of churn.
* **Computation Logic**: 
  $$\frac{\text{count}_{\text{Feb}} + 1}{\text{count}_{\text{Jan}} + 1}$$
* **Data Source**: `trx_2024-01.parquet`, `trx_2024-02.parquet`
* **Leakage Safety Check**: Relies entirely on completed historical months.

---

### 4.7 Engagement Features

#### Feature: `trx_type_diversity`
* **Definition**: The count of unique transaction types (P2P, BillPay, MerchantPay, CashIn, CashOut) utilized by the user.
* **Business Meaning**: Measure of multi-product adoption.
* **Why it Predicts Churn**: A user who only cashes out has low switching costs. A user who pays utility bills, pays merchants, and sends P2P money is deeply anchored in the wallet's ecosystem.
* **Computation Logic**: 
  $$\text{count}(\text{distinct}(\text{transaction\_type}))$$
* **Data Source**: `trx_2024-01.parquet`, `trx_2024-02.parquet`
* **Leakage Safety Check**: Bounded by the 2024-03-31 observation limit.

---

### 4.8 Sparse / Binary Features

#### Feature: `flag_zero_march_trx`
* **Definition**: A binary flag indicating whether the user had zero transaction activity during March (inferred from balance stability).
* **Business Meaning**: Flag for absolute dormancy during the final observation month.
* **Why it Predicts Churn**: If a user is completely dormant in March, the baseline probability that they will remain dormant in April is extremely high.
* **Computation Logic**: 
  $$\mathbb{I}(\text{zero\_balance\_days\_march} = 31 \quad \text{and} \quad \text{balance\_volatility} = 0)$$
* **Data Source**: `dayend_balance_march.parquet`
* **Leakage Safety Check**: Restricted to March data.

---

### 4.9 Trend Features

#### Feature: `balance_trend_march`
* **Definition**: The OLS linear regression slope of ending balances across the 31 days of March.
* **Business Meaning**: Represents the trajectory and velocity of wallet depletion.
* **Why it Predicts Churn**: A steep negative slope indicates the user is systematically emptying their wallet, which represents disengagement. A positive slope indicates accumulation and high retention probability.
* **Computation Logic**: 
  $$\text{Slope } (m) \text{ from } \text{Balance}_t = m \cdot t + c \quad \text{for } t \in [1, 31]$$
* **Data Source**: `dayend_balance_march.parquet`
* **Leakage Safety Check**: Relies on March daily ending balances.

---

## 5. Feature Computation Strategy

### 5.1 Data Processing Model
The pipeline uses a **batch-chunked** processing model rather than loading entire raw tables.
1. **Filtering**: The KYC table is loaded to build the master index of active `ACCOUNT_ID` values.
2. **Streaming Aggregation**: Transaction and balance files are streamed chunk-by-chunk. Each chunk updates running sums and counts in a temporary in-memory hash map.
3. **Write-Out**: Intermediate aggregates are written to parquet partition files, preventing memory overflows.

```
[Raw Parquets] ➔ [Chunk Streamer (Pandas/PyArrow)] ➔ [Incremental Aggregators] ➔ [Feature Store Parquets]
```

### 5.2 Memory Handling
* **PyArrow Columnar Filtering**: We only load necessary columns (e.g., `ACCOUNT_ID`, `amount`, `timestamp`) from Parquet files. Heavy columns are ignored at the I/O layer.
* **Chunking**: Large files are split into chunks of 500,000 rows.
* **Garbage Collection**: Dataframes are deleted from scope and `gc.collect()` is run explicitly between processing steps.

### 5.3 Join Strategy
* **Index-Based Joins**: Dataframes are set to use `ACCOUNT_ID` as a sorted index. Joins are executed as index-aligned mergers, which are orders of magnitude faster than standard hash joins.
* **Feature Store Integration**: Rather than executing one giant join, features are saved as independent parquet files in `./processed_data/` and joined sequentially at the very end of the pipeline.

---

## 6. Feature Quality Considerations

### Skewness Handling
Monetary transaction amounts and daily balances are highly right-skewed. To prevent extreme values from destabilizing model weights, we apply log-transformations:
$$\tilde{x} = \log(x + 1)$$
This stabilizes the variance and compresses the dynamic range of monetary metrics.

### Normalization Strategy
Standard standardizations (Z-score scaling) are calculated on the training folds:
$$z = \frac{x - \mu}{\sigma}$$
These scaling parameters ($\mu, \sigma$) are saved and applied to the validation/test folds to prevent information leakage during validation.

### Sparsity Handling
For users who do not perform a specific transaction type (e.g., zero merchant payments), their count and sum features default to `0.0` rather than `NaN`.

### Missing Value Strategy
Demographic variables (gender, region) containing missing data are imputed with the string `'UNKNOWN'`. This prevents the loss of records and allows the GBDT models to treat missingness as a distinct category.

---

## 7. Feature Risk Analysis

| Feature Group | Leakage Risk | Memory Cost | Compute Cost | Stability Risk | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Recency** | Low | Low | Medium | Low | Ensure the reference observation date is fixed to `2024-03-31`. |
| **Frequency** | Low | Medium | Low | Low | Pre-aggregate transaction files before joining. |
| **Monetary** | Low | Medium | Medium | Medium (Skew) | Apply $\log(x+1)$ scaling to address power-law distributions. |
| **Balance** | Low | High | High | Low | Read balance logs using PyArrow column filtering to save RAM. |
| **Trx Type Ratios** | Low | Medium | Medium | Medium | Handle division-by-zero errors when total outbound count is 0. |
| **Temporal Decay** | Low | Low | Low | Low | Use Laplace smoothing ($+1$) to avoid division by zero. |
| **Engagement** | Low | Low | Low | Low | Impute unrepresented categories as `'UNKNOWN'`. |
| **Trend (Slopes)** | Low | High | High | High | Use linear regression with fallback to `0.0` for static balances. |

---

## 8. Feature Summary Table

| Feature Name | Data Type | Raw Source Table | Extraction / Aggregation Logic | Business Meaning | Risk Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tenure_days` | Integer | `kyc.parquet` | Days from `ACCOUNT_OPEN_DATE` to `2024-03-31` | Account age / customer maturity | Low |
| `days_since_last_trx` | Float | `trx_2024-*.parquet` | Days from last transaction to `2024-03-31` | Length of wallet dormancy | Low |
| `out_count_total` | Integer | `trx_2024-*.parquet` | Total outbound transactions in Jan-Feb | Wallet transaction habit strength | Low |
| `out_sum_total` | Float | `trx_2024-*.parquet` | Sum of outbound transaction amounts in Jan-Feb | Total cash flow output | Medium |
| `final_balance_march` | Float | `dayend_balance_march.parquet`| Wallet ending balance on `2024-03-31` | Liquid capital remaining | Low |
| `balance_drop_march` | Float | `dayend_balance_march.parquet`| Balance on March 31 minus balance on March 1 | Magnitude of wallet draining | Medium |
| `balance_trend_march` | Float | `dayend_balance_march.parquet`| Linear slope of daily balances over March | Speed and direction of wallet empty | High |
| `zero_balance_days_march`| Integer | `dayend_balance_march.parquet`| Count of days with daily balance $< 10$ TK | Wallet abandonment indicator | Low |
| `trx_count_decay_jan_feb`| Float | `trx_2024-*.parquet` | (Feb Outbound + 1) / (Jan Outbound + 1) | Speed of transactional disengagement | Low |
| `trx_type_diversity` | Integer | `trx_2024-*.parquet` | Count of unique types used | Customer ecosystem integration | Low |
| `recency_pressure` | Float | Derived | `days_since_last_trx` / average transaction interval| Relative dormancy adjusted for habit | High |
| `digital_integration` | Float | Derived | (P2P + BillPay + MerchantPay) / Total Outbound | Digital wallet adoption level | Medium |

---

## 9. Production Considerations

### Scalability to 200M+ Rows
Aggregating large datasets requires distributed computing (like Spark) or highly optimized single-node engines (like DuckDB/Polars). The batch-streaming Pandas/PyArrow execution engine used here is optimized to handle high volumes on commodity hardware by keeping working memory under 8GB.

### Recomputation Costs
Feature generation is computationally expensive, especially slope calculations. To optimize this:
* Raw historical features (e.g., January/February transaction counts) are calculated once and cached.
* Only the rolling balance features (March balance metrics) are recomputed when new balance tables are published.

### Feature Store Design
In a production deployment, this feature store is divided into:
1. **Offline Store**: Parquet files stored on disk for model training.
2. **Online Store**: Key-value databases (e.g., Redis) that store pre-aggregated profile metrics (e.g., `final_balance`, `days_since_last_trx`) for sub-millisecond churn inference.

---

## 10. Final Notes

### Summary of Feature Strategy
The feature store focuses on capturing transactional disengagement (dormancy, frequency decay) and financial disengagement (wallet draining, zero-balance days). By capturing these features before March 31, we ensure high predictive power without relying on future information.

### Limitations
* **Missing March Transactions**: Because March transaction logs are missing, we cannot compute exact transaction-level recency or count features for the final month of the observation window.
* **Balance Frequency**: Balance logs are ending-day aggregates, which do not capture intra-day balance volatility.

### Assumptions
* Churn is defined by zero transaction activity in April. We assume that a customer who does not execute outbound or inbound transactions for 30 consecutive days is functionally churned.
* The KYC open date is assumed to be accurate and consistent across all regions.
