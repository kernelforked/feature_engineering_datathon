# Memory-Safe Pipeline Rewrite: FictiPay Churn Prediction System

> **Author**: Senior ML Systems Engineering Audit  
> **Scope**: Transform existing pandas-based pipeline into streaming + disk-first feature store architecture  
> **Constraint**: Zero changes to business logic, features, model stack, or evaluation strategy  

---

## Measured Dataset Dimensions (Actual)

| Table | File | Rows | Columns | Disk Size | Row Groups |
|-------|------|------|---------|-----------|------------|
| KYC | `kyc.parquet` | 1,000,000 | 5 | 6.8 MB | 1 |
| Transactions Jan | `trx_2024-01.parquet` | 25,385,709 | 6 | 530 MB | 34 |
| Transactions Feb | `trx_2024-02.parquet` | 23,373,303 | 6 | 489 MB | 34 |
| Transactions Mar | `trx_2024-03.parquet` | 24,522,366 | 6 | 512 MB | 34 |
| Balance Jan | `balance_2024-01.parquet` | 26,350,000 | 3 | 127 MB | 85 |
| Balance Feb | `balance_2024-02.parquet` | 24,650,000 | 3 | 100 MB | 85 |
| Balance Mar | `balance_2024-03.parquet` | 26,350,000 | 3 | 96 MB | 85 |

- **Train labels**: 595,000 accounts (519,565 active / 75,435 churned = 12.7% churn rate)
- **Test set**: 255,000 accounts
- **Total target IDs**: 850,000 unique Customer accounts
- **Total transaction rows**: 73,281,378 (~73M)
- **Total balance rows**: 77,350,000 (~77M)

---

## A. Current Memory Failure Points

### A.1 — Transaction File Triple-Scan (features.py)

The current pipeline reads each transaction parquet file **three independent times**:

| Pass | Function | Columns Loaded | Purpose |
|------|----------|----------------|---------|
| Pass 1 | `compute_recency()` | `SRC_ACCOUNT`, `DST_ACCOUNT`, `TRX_DATETIME` | Global max-datetime per account |
| Pass 2 | `compute_advanced_march_features()` | All 5 columns (March only) | Micro-window, velocity, type recency |
| Pass 3 | `aggregate_transactions_one_month()` | `SRC_ACCOUNT`, `DST_ACCOUNT`, `TRX_TYPE`, `TRX_AMT` | Monthly counts/sums/stats |

**Problem**: Each `pd.read_parquet()` call materializes the entire monthly file into RAM. A single transaction file decompresses to approximately **2.5–3.0 GB** in pandas DataFrame form (25M rows × 6 columns with string types). Three passes across three files means the pipeline attempts to allocate **~9 GB per scan cycle**, totaling **~27 GB of cumulative RAM allocation** before GC can reclaim anything.

**Specific code locations**:
- `features.py:38` → `pd.read_parquet(filepath, columns=[...])` in `compute_recency()`
- `features.py:78` → `pd.read_parquet(filepath, columns=[...])` in `compute_advanced_march_features()`
- `features.py:182` → `pd.read_parquet(filepath, columns=[...])` in `aggregate_transactions_one_month()`

### A.2 — Balance File Full-Load Before Chunking (features.py)

In `aggregate_balances()` at line 329:
```python
df = pd.read_parquet(filepath, columns=["ACCOUNT_ID", "DATE", "AVAILABLE_BALANCE"])
df = df[df["ACCOUNT_ID"].isin(target_ids)]
```

The **entire** monthly balance file (~26M rows) is loaded into RAM **before** the `isin()` filter is applied. Even after filtering to 850K target IDs, the initial load requires approximately **1.5 GB per month** (26M rows × 3 columns). The subsequent chunked pivot at line 362 is effective only **after** the full file is already in memory — the chunking does not prevent the initial OOM.

### A.3 — Mega-Join Explosion (features.py)

At line 490:
```python
features = kyc_df.join(trx_df, how="left").join(bal_df, how="left").fillna(0)
```

This creates a single pandas DataFrame of 850,000 rows × ~130+ columns. Each intermediate `.join()` creates a temporary copy. With string-based `ACCOUNT_ID` indexes, pandas performs hash-table joins that temporarily hold **both** the original and result DataFrames simultaneously. Peak memory at this line: approximately **3× the final DataFrame size** = ~3 GB for intermediate join buffers.

### A.4 — Full Feature Matrix in Training RAM (train.py)

At line 33–34:
```python
train_df = pd.read_parquet("./processed_data/train_features.parquet")
test_df = pd.read_parquet("./processed_data/test_features.parquet")
```

Both DataFrames are loaded simultaneously into RAM and kept alive throughout the entire 10-fold CV training loop. The training DataFrame alone is 595,000 × 130+ columns ≈ **600 MB**. During each fold, `X_train` and `X_val` slices are created via `.iloc[]` which creates **views** (safe), but `model.fit()` in XGBoost internally creates a `DMatrix` copy, and LightGBM creates a `Dataset` copy — effectively tripling the RAM footprint for each fold iteration.

### A.5 — Preprocessing Leakage & Global Fit (train.py)

At lines 361–365:
```python
X_imputed = imputer.fit_transform(X)          # fits on ALL data including val folds
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), ...)  # fits on ALL data
```

This creates **two full-size copies** of the feature matrix (`X_imputed` and `X_scaled`), each ~600 MB. Combined with the original `X`, this is **~1.8 GB** just for the linear model preprocessing step. Additionally, fitting the imputer and scaler on the entire dataset before CV creates validation leakage.

### A.6 — Python Dict Accumulation in Recency (features.py)

At lines 31–55, `compute_recency()` accumulates a Python dict `last_trx = {}` via a row-by-row loop:
```python
for acct, dt in src_last.items():
    if acct not in last_trx or dt > last_trx[acct]:
        last_trx[acct] = dt
```

For 850,000 accounts with string keys, this dict consumes approximately **200 MB** of heap memory due to Python object overhead (each string key ~100 bytes, each Timestamp value ~80 bytes). This is unnecessary — the same operation can be done via a single `groupby().max()` merge.

---

## B. Streaming Transformation Design

### B.1 — Core Architectural Principle: Single-Pass Row-Group Streaming

Parquet files are internally divided into **row groups** (the transaction files have 34 row groups each; balance files have 85 each). PyArrow's `ParquetFile.iter_batches()` can read one row group at a time without loading the full file. This is the foundation of the streaming transformation.

**Key API**:
```python
import pyarrow.parquet as pq

pf = pq.ParquetFile("trx_2024-01.parquet")
for batch in pf.iter_batches(batch_size=1_000_000, columns=["SRC_ACCOUNT", "TRX_AMT"]):
    chunk_df = batch.to_pandas()
    # Process chunk, aggregate, discard
```

Each batch loads approximately **750K rows** (one row group ÷ column subset), consuming roughly **50–100 MB** of RAM. After processing, the chunk is discarded and GC reclaims the memory before the next batch arrives.

### B.2 — Single-Pass Multi-Accumulator Pattern

The core transformation eliminates the triple-scan problem by processing each transaction file **exactly once**. During that single pass, multiple accumulators simultaneously collect data for different feature groups:

```
For each row-group batch in a transaction file:
    ├─ Accumulator 1: Update recency max-timestamps
    ├─ Accumulator 2: Update outbound count/sum/mean/std/max
    ├─ Accumulator 3: Update inbound count/sum
    ├─ Accumulator 4: Update type-specific counts
    ├─ Accumulator 5: (March only) Update micro-window 7d/14d stats
    └─ Accumulator 6: (March only) Update type-specific recency
```

All accumulators operate on **aggregated DataFrames indexed by ACCOUNT_ID** (850K rows max), which are ~50 MB each. The raw batch is discarded after each iteration.

### B.3 — Disk-First Intermediate Storage

Every feature group is written to disk as a separate parquet file immediately after computation:

```
feature_store/
├── kyc_features.parquet          (~850K × 25 cols)
├── trx_recency.parquet           (~850K × 1 col)
├── trx_agg_jan.parquet           (~850K × 15 cols)
├── trx_agg_feb.parquet           (~850K × 15 cols)
├── trx_agg_march.parquet         (~850K × 15 cols)
├── trx_march_advanced.parquet    (~850K × 20 cols)
├── trx_cross_month.parquet       (~850K × 15 cols)
├── bal_agg_jan.parquet           (~850K × 4 cols)
├── bal_agg_feb.parquet           (~850K × 4 cols)
├── bal_agg_march.parquet         (~850K × 4 cols)
├── bal_march_advanced.parquet    (~850K × 10 cols)
├── abstract_features.parquet     (~850K × 6 cols)
├── sparsity_flags.parquet        (~850K × 4 cols)
└── log_transforms.parquet        (~850K × 16 cols)
```

Each file is **< 30 MB on disk**. The final join reads these files sequentially and merges them in a staged pipeline — never holding more than two feature groups in RAM simultaneously.

### B.4 — ACCOUNT_ID Optimization

Before any processing begins, convert `ACCOUNT_ID` from variable-length strings to a **categorical / integer mapping**:

```python
target_ids_list = sorted(target_ids)
id_to_int = {aid: i for i, aid in enumerate(target_ids_list)}
```

This reduces per-key memory from ~100 bytes (Python string) to 8 bytes (int64) — a **12× memory reduction** on all join indexes and groupby keys.

Apply this mapping immediately when reading each batch:
```python
chunk_df["ACCOUNT_ID"] = chunk_df["SRC_ACCOUNT"].map(id_to_int)
```

Non-target accounts (merchants, billers) map to `NaN` and are dropped, which simultaneously performs the `isin()` filter without creating a separate set lookup.

---

## C. Feature Engineering Rewrite (Streaming Version)

### C.1 — KYC Features (Unchanged — Already Safe)

**Current behavior**: Loads `kyc.parquet` (1M rows, 6.8 MB on disk), filters to 850K target IDs, computes `tenure_days`, one-hot encodes `GENDER` and `REGION`.

**Memory footprint**: ~50 MB in RAM. This is already safe.

**Streaming change**: None required. Write output to `feature_store/kyc_features.parquet` and release from RAM.

### C.2 — Transaction Recency Features (Streaming Max-Timestamp)

**Current**: `compute_recency()` — 3 separate file reads, Python dict accumulation.

**Streaming rewrite**:

```python
# Initialize accumulator as a pandas Series indexed by integer ACCOUNT_ID
global_max_dt = pd.Series(pd.NaT, index=range(len(target_ids_list)), dtype="datetime64[ns]")

for filepath in trx_files:
    pf = pq.ParquetFile(filepath)
    for batch in pf.iter_batches(columns=["SRC_ACCOUNT", "DST_ACCOUNT", "TRX_DATETIME"]):
        chunk = batch.to_pandas()
        chunk["TRX_DATETIME"] = pd.to_datetime(chunk["TRX_DATETIME"])
        
        # Source side
        chunk["src_int"] = chunk["SRC_ACCOUNT"].map(id_to_int)
        src_valid = chunk.dropna(subset=["src_int"])
        src_max = src_valid.groupby("src_int")["TRX_DATETIME"].max()
        global_max_dt.update(src_max[src_max > global_max_dt.reindex(src_max.index)])
        
        # Destination side
        chunk["dst_int"] = chunk["DST_ACCOUNT"].map(id_to_int)
        dst_valid = chunk.dropna(subset=["dst_int"])
        dst_max = dst_valid.groupby("dst_int")["TRX_DATETIME"].max()
        global_max_dt.update(dst_max[dst_max > global_max_dt.reindex(dst_max.index)])
        
        del chunk
```

**Memory**: Accumulator = 850K × 8 bytes = ~7 MB. Each batch = ~50 MB. **Peak: ~60 MB**.

**Output**: `feature_store/trx_recency.parquet` containing `days_since_last_trx`.

### C.3 — Monthly Transaction Aggregation (Streaming Counters)

**Current**: `aggregate_transactions_one_month()` loads full month into RAM, groupby, then discards.

**Streaming rewrite**: Use Welford's online algorithm for streaming mean/std/max:

```python
# Per-month accumulators (all indexed by int ACCOUNT_ID, 850K rows)
out_count = pd.Series(0, index=target_int_ids, dtype="int64")
out_sum   = pd.Series(0.0, index=target_int_ids, dtype="float64")
out_m2    = pd.Series(0.0, index=target_int_ids, dtype="float64")  # for Welford std
out_max   = pd.Series(-np.inf, index=target_int_ids, dtype="float64")
in_count  = pd.Series(0, index=target_int_ids, dtype="int64")
in_sum    = pd.Series(0.0, index=target_int_ids, dtype="float64")
type_counts = pd.DataFrame(0, index=target_int_ids, 
                           columns=["P2P","MerchantPay","BillPay","CashIn","CashOut"], dtype="int64")

pf = pq.ParquetFile(filepath)
for batch in pf.iter_batches(columns=["SRC_ACCOUNT","DST_ACCOUNT","TRX_TYPE","TRX_AMT"]):
    chunk = batch.to_pandas()
    
    # --- Outbound ---
    chunk["src_int"] = chunk["SRC_ACCOUNT"].map(id_to_int)
    src = chunk.dropna(subset=["src_int"])
    
    grp = src.groupby("src_int")
    batch_count = grp["TRX_AMT"].count()
    batch_sum   = grp["TRX_AMT"].sum()
    batch_max   = grp["TRX_AMT"].max()
    
    # Welford update for streaming std
    old_count = out_count.reindex(batch_count.index)
    old_mean  = out_sum.reindex(batch_count.index) / (old_count + 1e-10)
    new_count = old_count + batch_count
    new_mean  = (out_sum.reindex(batch_count.index) + batch_sum) / (new_count + 1e-10)
    # M2 update: M2_new = M2_old + batch_M2 + delta^2 * old_count * batch_count / new_count
    batch_mean = batch_sum / (batch_count + 1e-10)
    delta = batch_mean - old_mean
    batch_m2 = grp["TRX_AMT"].var() * (batch_count - 1)  # sample var to sum-of-squares
    out_m2.update(out_m2.reindex(batch_count.index) + batch_m2.fillna(0) + 
                  delta**2 * old_count * batch_count / (new_count + 1e-10))
    
    out_count.update(new_count)
    out_sum.update(out_sum.reindex(batch_count.index) + batch_sum)
    out_max.update(batch_max[batch_max > out_max.reindex(batch_max.index)])
    
    # Type counts
    type_grp = src.groupby(["src_int", "TRX_TYPE"]).size().unstack(fill_value=0)
    for col in type_grp.columns:
        if col in type_counts.columns:
            type_counts[col].update(type_counts[col].reindex(type_grp.index) + type_grp[col])
    
    # --- Inbound ---
    chunk["dst_int"] = chunk["DST_ACCOUNT"].map(id_to_int)
    dst = chunk.dropna(subset=["dst_int"])
    dst_grp = dst.groupby("dst_int")
    in_count.update(in_count.reindex(dst_grp.ngroups) + dst_grp["TRX_AMT"].count())
    in_sum.update(in_sum.reindex(dst_grp.ngroups) + dst_grp["TRX_AMT"].sum())
    
    del chunk
```

**Memory**: 10 accumulators × 850K × 8 bytes = ~68 MB total. Each batch = ~50 MB. **Peak: ~120 MB**.

**Output**: `feature_store/trx_agg_{month}.parquet` containing `out_count`, `out_sum`, `out_avg`, `out_std`, `out_max`, `in_count`, `in_sum`, `count_{Type}` per month.

### C.4 — Advanced March Features (Integrated into Single March Pass)

**Current**: `compute_advanced_march_features()` loads the entire March file a second time.

**Streaming rewrite**: Merge into the March monthly aggregation pass. During the March batch iteration, add simultaneous accumulators for:

- **Directional recency** (outbound/inbound max datetime)
- **Type-specific recency** (per-type max datetime)
- **Micro-window 7d/14d** (filter `TRX_DATETIME >= 2024-03-25` and `>= 2024-03-18` within each batch)
- **Velocity ratios** (computed post-accumulation from micro-window / total March counts)

This requires loading `TRX_DATETIME` only for the March file (adding one extra column to the March batch read). January and February scans remain datetime-free.

**Additional accumulators for March**:
```python
# March-only accumulators
out_last_dt = pd.Series(pd.NaT, index=target_int_ids, dtype="datetime64[ns]")
in_last_dt  = pd.Series(pd.NaT, index=target_int_ids, dtype="datetime64[ns]")
type_last_dt = {t: pd.Series(pd.NaT, index=target_int_ids, dtype="datetime64[ns]") 
                for t in ["P2P","MerchantPay","BillPay","CashOut","CashIn"]}
out_7d_count = pd.Series(0, index=target_int_ids, dtype="int64")
out_7d_sum   = pd.Series(0.0, index=target_int_ids, dtype="float64")
out_14d_count = pd.Series(0, index=target_int_ids, dtype="int64")
out_14d_sum   = pd.Series(0.0, index=target_int_ids, dtype="float64")
in_7d_count  = pd.Series(0, index=target_int_ids, dtype="int64")
in_7d_sum    = pd.Series(0.0, index=target_int_ids, dtype="float64")
in_14d_count = pd.Series(0, index=target_int_ids, dtype="int64")
in_14d_sum   = pd.Series(0.0, index=target_int_ids, dtype="float64")
```

**Memory**: 18 additional accumulators × 850K × 8 bytes = ~120 MB. **Combined March peak: ~240 MB** (all accumulators + one batch).

**Output**: `feature_store/trx_march_advanced.parquet`.

### C.5 — Balance Features (Row-Group Streaming with Chunked Pivot)

**Current**: `aggregate_balances()` loads the entire monthly balance file (~26M rows) into RAM before filtering.

**Streaming rewrite**: Use `iter_batches()` to stream balance data:

```python
# Monthly stat accumulators (Welford online for mean/std)
bal_count = pd.Series(0, index=target_int_ids, dtype="int64")
bal_sum   = pd.Series(0.0, index=target_int_ids, dtype="float64")
bal_m2    = pd.Series(0.0, index=target_int_ids, dtype="float64")
bal_min   = pd.Series(np.inf, index=target_int_ids, dtype="float64")
bal_max   = pd.Series(-np.inf, index=target_int_ids, dtype="float64")

pf = pq.ParquetFile(filepath)
for batch in pf.iter_batches(columns=["ACCOUNT_ID", "AVAILABLE_BALANCE"]):
    chunk = batch.to_pandas()
    chunk["aid_int"] = chunk["ACCOUNT_ID"].map(id_to_int)
    chunk = chunk.dropna(subset=["aid_int"])
    
    grp = chunk.groupby("aid_int")["AVAILABLE_BALANCE"]
    # Online accumulation (same Welford pattern as C.3)
    ...
```

For March specifically, the pivot-based features (daily slope, final balance, zero-balance days) require **day-level granularity**. The solution is a **two-pass approach within a single file read**:

**March Pass 1** (streaming): Compute `mean_bal`, `std_bal`, `min_bal`, `max_bal` using online accumulators.

**March Pass 2** (chunked by account): Instead of loading the entire March balance file and pivoting, use a **row-group accumulator dictionary** that collects daily balances per account chunk:

```python
# Pre-allocate a dense matrix for march daily balances: 850K accounts × 31 days
# At float32: 850K × 31 × 4 bytes = ~105 MB (fits comfortably in RAM)
daily_bal = np.zeros((len(target_int_ids), 31), dtype=np.float32)

pf = pq.ParquetFile(march_balance_path)
for batch in pf.iter_batches(columns=["ACCOUNT_ID", "DATE", "AVAILABLE_BALANCE"]):
    chunk = batch.to_pandas()
    chunk["aid_int"] = chunk["ACCOUNT_ID"].map(id_to_int)
    chunk = chunk.dropna(subset=["aid_int"])
    chunk["day"] = pd.to_datetime(chunk["DATE"]).dt.day - 1  # 0-indexed
    
    # Scatter values into the dense matrix
    rows = chunk["aid_int"].astype(int).values
    cols = chunk["day"].values
    vals = chunk["AVAILABLE_BALANCE"].values.astype(np.float32)
    daily_bal[rows, cols] = vals
```

Then compute all derived features vectorially on the dense matrix:
```python
final_balance = daily_bal[:, 30]           # day 31
balance_drop  = daily_bal[:, 30] - daily_bal[:, 0]  # day 31 - day 1
# Linear trend slope
t_centered = np.arange(31) - 15.0
denom = (t_centered ** 2).sum()
balance_trend = daily_bal @ (t_centered / denom)
# Zero balance days
zero_days = (daily_bal < 10.0).sum(axis=1)
# Micro-window features
mean_bal_7d = daily_bal[:, 24:31].mean(axis=1)   # days 25-31
mean_bal_14d = daily_bal[:, 17:31].mean(axis=1)  # days 18-31
zero_days_7d = (daily_bal[:, 24:31] < 10.0).sum(axis=1)
# 14d trend
t_14 = np.arange(14) - 6.5
denom_14 = (t_14 ** 2).sum()
bal_trend_14d = daily_bal[:, 17:31] @ (t_14 / denom_14)
```

**Memory**: Dense matrix = 105 MB. Accumulators = ~35 MB. Batch = ~30 MB. **Peak: ~170 MB**.

**Output**: `feature_store/bal_agg_{month}.parquet` and `feature_store/bal_march_advanced.parquet`.

### C.6 — Cross-Month Derived Features (Post-Aggregation, Disk-Based)

Decay features, activity share, diversity, ratios, net flow, sparsity flags, abstract features, and log transforms are all computed on the **already-aggregated feature store files**. Each file is 850K rows × <20 columns = ~50 MB.

**Streaming rewrite**: Load only the required feature store files, compute derived features, write to disk, release:

```python
# Load only what's needed
trx_jan = pd.read_parquet("feature_store/trx_agg_jan.parquet")
trx_feb = pd.read_parquet("feature_store/trx_agg_feb.parquet")
trx_mar = pd.read_parquet("feature_store/trx_agg_march.parquet")

# Decay features
cross = pd.DataFrame(index=trx_jan.index)
cross["trx_count_decay_jan_feb"] = (trx_feb["out_count"] - trx_jan["out_count"]) / (trx_jan["out_count"] + 1)
cross["trx_count_decay_feb_march"] = (trx_mar["out_count"] - trx_feb["out_count"]) / (trx_feb["out_count"] + 1)
cross["trx_count_decay_jan_march"] = (trx_mar["out_count"] - trx_jan["out_count"]) / (trx_jan["out_count"] + 1)

# Totals
cross["out_count_total"] = trx_jan["out_count"] + trx_feb["out_count"] + trx_mar["out_count"]
# ... (all other cross-month features)

cross.to_parquet("feature_store/trx_cross_month.parquet")
del trx_jan, trx_feb, trx_mar, cross
gc.collect()
```

**Memory**: 3 monthly files (~50 MB each) + result (~30 MB) = **~180 MB peak**.

### C.7 — Abstract Features, Sparsity Flags, Log Transforms

Computed identically to current logic, but reading from the feature store files rather than from a mega-DataFrame. Each computation loads only the specific columns it needs:

```python
# Example: recency_pressure needs days_since_last_trx and out_count_total
recency = pd.read_parquet("feature_store/trx_recency.parquet")
cross = pd.read_parquet("feature_store/trx_cross_month.parquet", columns=["out_count_total"])
ati = 90.0 / (cross["out_count_total"] + 1)
abstract = pd.DataFrame(index=recency.index)
abstract["recency_pressure"] = recency["days_since_last_trx"] / (ati + 1e-5)
# ... (all other abstract features)
abstract.to_parquet("feature_store/abstract_features.parquet")
```

**Memory**: ~20 MB per computation step. **Peak: ~50 MB**.

---

## D. Join Optimization Strategy

### D.1 — Staged Columnar Join Pipeline

**Current failure**: Single-line `kyc_df.join(trx_df, how="left").join(bal_df, how="left")` creates massive intermediate copies.

**Rewrite**: Sequential two-file merges with immediate disk write:

```python
feature_store_files = [
    "feature_store/kyc_features.parquet",
    "feature_store/trx_recency.parquet",
    "feature_store/trx_agg_jan.parquet",
    "feature_store/trx_agg_feb.parquet",
    "feature_store/trx_agg_march.parquet",
    "feature_store/trx_march_advanced.parquet",
    "feature_store/trx_cross_month.parquet",
    "feature_store/bal_agg_jan.parquet",
    "feature_store/bal_agg_feb.parquet",
    "feature_store/bal_agg_march.parquet",
    "feature_store/bal_march_advanced.parquet",
    "feature_store/abstract_features.parquet",
    "feature_store/sparsity_flags.parquet",
    "feature_store/log_transforms.parquet",
]

# Build final table by concatenating columns (all share same int index)
final = pd.read_parquet(feature_store_files[0])
for fpath in feature_store_files[1:]:
    part = pd.read_parquet(fpath)
    final = final.join(part, how="left")
    del part
    gc.collect()

final.fillna(0, inplace=True)
```

**Why this is safe**: Each partial file is < 30 MB. The running `final` DataFrame grows incrementally. At no point is there a 3× memory spike from temporary copies because we join one small file at a time and immediately discard the input.

### D.2 — Integer Index Joins

All feature store files use the same integer index (`aid_int` mapped from `ACCOUNT_ID`). Integer joins in pandas are **5–10× faster** and use **12× less memory** than string joins. The mapping back to string `ACCOUNT_ID` happens only at the final output stage:

```python
final["ACCOUNT_ID"] = final.index.map(int_to_id)  # reverse lookup
```

### D.3 — Train/Test Split via Index Masking

Instead of creating separate DataFrames and merging labels:

```python
train_mask = final["ACCOUNT_ID"].isin(train_ids_set)
test_mask  = final["ACCOUNT_ID"].isin(test_ids_set)

# Write directly to disk — never hold both in RAM simultaneously
final[train_mask].to_parquet("processed_data/train_features.parquet")
final[test_mask].to_parquet("processed_data/test_features.parquet")
del final
gc.collect()
```

---

## E. Model Training Memory Fix

### E.1 — Load from Feature Store Only

```python
train_df = pd.read_parquet("./processed_data/train_features.parquet")
# NO raw data files are opened during training
# Memory: 595K × 130 cols × 8 bytes ≈ 600 MB (acceptable for training)
```

### E.2 — Fix Scaler/Imputer Leakage (Fold-Internal Fitting)

**Current bug** (train.py lines 361–365): Imputer and Scaler are fit on the entire dataset before CV splitting.

**Fix**: Move fit inside the CV loop:

```python
def train_linear_models(X, y, X_test, cv):
    lr_oof = np.zeros(len(X))
    lr_test = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_raw = X.iloc[train_idx]
        X_val_raw   = X.iloc[val_idx]
        
        # Fit imputer and scaler ONLY on training fold
        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_val_imp   = imputer.transform(X_val_raw)
        X_test_imp  = imputer.transform(X_test)
        
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_val_scaled   = scaler.transform(X_val_imp)
        X_test_scaled  = scaler.transform(X_test_imp)
        
        # Train LR on this fold's scaled data
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_train_scaled, y.iloc[train_idx])
        
        lr_oof[val_idx] = model.predict_proba(X_val_scaled)[:, 1]
        lr_test += model.predict_proba(X_test_scaled)[:, 1] / cv.n_splits
        
        del X_train_imp, X_val_imp, X_train_scaled, X_val_scaled
        gc.collect()
```

**Memory saving**: Eliminates the two full-dataset copies (`X_imputed`, `X_scaled`). Each fold only holds the training portion (~540K rows) and validation portion (~60K rows). **Saves ~1.2 GB** of peak RAM.

### E.3 — Explicit Memory Cleanup Between Models

```python
# After each model completes, release model array copies
lgb_oof, lgb_test, lgb_models = train_lgb(X, y, X_test, cv)
gc.collect()

# Release X_test_dmatrix internal copies
xgb_oof, xgb_test, xgb_models = train_xgb(X, y, X_test, cv)
gc.collect()

# Keep only fold-0 model for SHAP, discard the rest
lgb_model_for_shap = lgb_models[0]
del lgb_models[1:]
```

### E.4 — Optional: Fold-Based Disk Spilling for Extreme Memory Constraints

If even the 600 MB training DataFrame is too large, use index-based loading:

```python
# Pre-save fold indices
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    np.save(f"folds/fold_{fold}_train.npy", train_idx)
    np.save(f"folds/fold_{fold}_val.npy", val_idx)

# During training, load only what's needed
train_idx = np.load(f"folds/fold_{fold}_train.npy")
X_train = pd.read_parquet("processed_data/train_features.parquet").iloc[train_idx]
```

This is only needed on machines with < 4 GB available RAM.

---

## F. Safe Pipeline Architecture (Final State)

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA (on disk)                       │
│  kyc.parquet (7MB) | trx_*.parquet (1.5GB) | bal_*.parquet (323MB)│
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: STREAM PROCESSING (lazy scan)              │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ PyArrow iter_batches() — one row-group at a time    │        │
│  │ Peak RAM per batch: ~50-100 MB                      │        │
│  │ Single-pass multi-accumulator pattern               │        │
│  │ Integer ACCOUNT_ID mapping                          │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                 │
│  Transaction files: 1 pass per file (3 total)                   │
│  Balance files: 1 pass per file for stats (3 total)             │
│                + 1 pass for March daily matrix                  │
│  Total file reads: 7 (down from 12 in current pipeline)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│           STEP 2: FEATURE STORE (parquet partitions)            │
│                                                                 │
│  feature_store/                                                 │
│  ├── kyc_features.parquet           (850K × 25)   ~10 MB       │
│  ├── trx_recency.parquet            (850K × 1)    ~3 MB        │
│  ├── trx_agg_{jan,feb,march}.parquet (850K × 15)  ~20 MB each │
│  ├── trx_march_advanced.parquet     (850K × 20)   ~25 MB       │
│  ├── trx_cross_month.parquet        (850K × 15)   ~20 MB       │
│  ├── bal_agg_{jan,feb,march}.parquet (850K × 4)   ~5 MB each  │
│  ├── bal_march_advanced.parquet     (850K × 10)   ~12 MB       │
│  ├── abstract_features.parquet      (850K × 6)    ~8 MB        │
│  ├── sparsity_flags.parquet         (850K × 4)    ~4 MB        │
│  └── log_transforms.parquet         (850K × 16)   ~18 MB       │
│                                                                 │
│  Total feature store disk: ~190 MB                              │
│  Peak RAM during creation: ~250 MB                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│          STEP 3: STAGED JOIN → TRAINING TABLE (compressed)      │
│                                                                 │
│  Sequential column-merge of 14 parquet files                    │
│  Integer-indexed joins (no string keys)                         │
│  Incremental join with immediate cleanup                        │
│                                                                 │
│  Output:                                                        │
│  ├── processed_data/train_features.parquet (595K × 130+) ~400MB│
│  └── processed_data/test_features.parquet  (255K × 130+) ~170MB│
│                                                                 │
│  Peak RAM: ~800 MB (final table + one partial file)             │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│       STEP 4: MODEL TRAINING (LightGBM / XGBoost / etc.)       │
│                                                                 │
│  Load train_features.parquet (~600 MB)                          │
│  10-Fold Stratified CV with fold-internal scaling               │
│  Sequential model zoo execution with inter-model GC             │
│                                                                 │
│  Models: LGB → XGB → CatBoost → RF → LR → MLP                 │
│  Peak RAM: ~2 GB (feature matrix + model internal buffers)      │
│                                                                 │
│  Output: predictions/oof_predictions.parquet                    │
│          predictions/test_predictions.parquet                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 5: ENSEMBLE + PREDICTIONS                 │
│                                                                 │
│  Load OOF + test predictions (~50 MB total)                     │
│  Stacking meta-learner + Rank-Average Blend                     │
│  Logistic calibration + cost-sensitive thresholding             │
│                                                                 │
│  Output: predictions.csv                                        │
│  Peak RAM: ~100 MB                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Total pipeline peak RAM**: ~2 GB (during model training)  
**Current pipeline peak RAM**: ~12–15 GB (during feature engineering joins)  
**Reduction**: **~85% memory reduction**

---

## G. Memory Safety Classification

| Pipeline Stage | Current Status | Rewritten Status | Peak RAM (Current) | Peak RAM (Rewritten) |
|---------------|---------------|-----------------|-------------------|---------------------|
| KYC Load & Encode | SAFE | SAFE | ~50 MB | ~50 MB |
| Transaction Recency Scan | BROKEN | SAFE | ~3 GB per file | ~60 MB |
| Transaction Monthly Aggregation | BROKEN | SAFE | ~3 GB per file | ~120 MB |
| March Advanced Features | BROKEN | SAFE | ~3 GB | ~240 MB (merged into single pass) |
| Balance Monthly Stats | RISKY | SAFE | ~1.5 GB per file | ~100 MB |
| Balance March Pivot (daily) | RISKY | SAFE | ~1.5 GB + pivot | ~170 MB (dense matrix) |
| Cross-Month Derived Features | SAFE | SAFE | ~200 MB | ~180 MB |
| Abstract / Flags / LogTransforms | SAFE | SAFE | ~100 MB | ~50 MB |
| Mega-Join (4-way) | BROKEN | SAFE | ~3 GB (3× copies) | ~800 MB (staged) |
| Training Data Load | RISKY | SAFE | ~1.2 GB (train+test) | ~600 MB (train only) |
| Scaler/Imputer Preprocessing | BROKEN (leakage) | SAFE | ~1.8 GB | ~600 MB (fold-internal) |
| Model Training (LGB/XGB) | RISKY | SAFE | ~2.5 GB | ~2 GB |
| Model Training (RF/MLP) | RISKY | SAFE | ~2.5 GB | ~2 GB |
| Ensemble + Submission | SAFE | SAFE | ~100 MB | ~100 MB |

---

## H. Final Output: Rewritten Pipeline (Exact Execution Order)

### Step 0: Initialize

```
0.1  Load train_labels.csv and test.csv (850K IDs total)
0.2  Build sorted ACCOUNT_ID → integer mapping (id_to_int, int_to_id)
0.3  Create feature_store/ directory
0.4  Set reference date = 2024-03-31
```
**Memory**: ~50 MB  
**Classification**: SAFE

---

### Step 1: KYC Feature Extraction

```
1.1  Read kyc.parquet (1M rows, 7 MB)
1.2  Filter to 850K target IDs
1.3  Compute tenure_days
1.4  One-hot encode GENDER, REGION
1.5  Map ACCOUNT_ID to integer index
1.6  Write → feature_store/kyc_features.parquet
1.7  Release from RAM
```
**Memory**: ~50 MB  
**Classification**: SAFE

---

### Step 2: Transaction Feature Extraction (Single-Pass Per File)

```
FOR each month in [jan (trx_2024-01), feb (trx_2024-02), march (trx_2024-03)]:

  2.1  Initialize accumulators:
       - out_count, out_sum, out_m2, out_max (for streaming mean/std/max)
       - in_count, in_sum
       - type_counts (5-column DataFrame)
       - IF march: global_max_dt, out_last_dt, in_last_dt, type_last_dt[5]
       - IF march: micro-window accumulators (7d/14d count/sum, in + out)

  2.2  Open ParquetFile, iterate row-group batches:
       - Columns: SRC_ACCOUNT, DST_ACCOUNT, TRX_TYPE, TRX_AMT
       - IF march: also TRX_DATETIME
       - Batch size: ~750K rows (~50-80 MB)

  2.3  For each batch:
       a) Map SRC_ACCOUNT/DST_ACCOUNT → integer IDs, drop non-target rows
       b) Update outbound accumulators (Welford online for std)
       c) Update inbound accumulators
       d) Update type-specific counters
       e) IF march:
          - Update directional recency max-timestamps
          - Update type-specific recency max-timestamps
          - Filter to 7d/14d windows, update micro-window accumulators
       f) Update global recency max-timestamp (all months)
       g) Discard batch, gc.collect()

  2.4  Finalize accumulators:
       - out_avg = out_sum / (out_count + 1e-10)
       - out_std = sqrt(out_m2 / (out_count - 1 + 1e-10))
       - IF march: recency days = (ref_date - max_timestamps).days

  2.5  Write → feature_store/trx_agg_{month}.parquet
  2.6  IF march: Write → feature_store/trx_march_advanced.parquet
  2.7  Release accumulators from RAM
```

```
AFTER all three months:

  2.8  Write → feature_store/trx_recency.parquet (global max datetime → days)
```

**Memory per file**: ~120–240 MB (March is highest due to extra accumulators)  
**Total file reads**: 3 (one per month) — **down from 9 in current pipeline**  
**Classification**: SAFE

---

### Step 3: Balance Feature Extraction (Streaming)

```
FOR each month in [jan, feb, march]:

  3.1  Initialize accumulators: bal_count, bal_sum, bal_m2, bal_min, bal_max

  3.2  Open ParquetFile, iterate row-group batches:
       - Columns: ACCOUNT_ID, AVAILABLE_BALANCE (and DATE for March)
       - Batch size: ~310K rows (~30 MB)

  3.3  For each batch:
       a) Map ACCOUNT_ID → integer, drop non-target
       b) Update Welford online accumulators for mean/std/min/max
       c) IF march: scatter AVAILABLE_BALANCE into dense daily matrix
          (850K × 31 float32 = ~105 MB, allocated once)
       d) Discard batch

  3.4  Finalize:
       - mean_bal = bal_sum / (bal_count + 1e-10)
       - std_bal  = sqrt(bal_m2 / (bal_count - 1 + 1e-10))

  3.5  IF march:
       - Compute from dense matrix: final_balance, balance_drop,
         balance_trend (31d slope), zero_balance_days,
         mean_bal_7d, mean_bal_14d, zero_days_7d, bal_trend_14d
       - Write → feature_store/bal_march_advanced.parquet
       - Release dense matrix

  3.6  Write → feature_store/bal_agg_{month}.parquet
  3.7  Release accumulators
```

**Memory**: ~170 MB peak (March, due to dense matrix)  
**Total file reads**: 3 (one per month — same as current, but streaming)  
**Classification**: SAFE

---

### Step 4: Cross-Month Derived Features

```
4.1  Load trx_agg_jan, trx_agg_feb, trx_agg_march from feature store
4.2  Compute:
     - out_count_total, out_sum_total, in_count_total, in_sum_total
     - trx_count_decay_jan_feb, _feb_march, _jan_march
     - march_activity_share
     - count_{Type}_total (sum across months)
     - trx_type_diversity
     - merchant_pay_ratio, bill_pay_ratio, p2p_ratio, cashout_ratio
     - net_flow_march
4.3  Write → feature_store/trx_cross_month.parquet
4.4  Release monthly files from RAM
```

**Memory**: ~180 MB  
**Classification**: SAFE

---

### Step 5: Abstract Features, Sparsity Flags, Log Transforms

```
5.1  Load required columns from feature store files:
     - trx_recency, trx_cross_month, trx_march_advanced,
       bal_agg_march, bal_march_advanced

5.2  Compute abstract features:
     - recency_pressure, digital_integration, depletion_velocity,
       store_of_value, network_stickiness, wallet_strain
5.3  Write → feature_store/abstract_features.parquet

5.4  Compute sparsity flags:
     - flag_zero_bill_pay, flag_zero_merchant_pay,
       flag_zero_march_trx, flag_zero_trx_last_7d_march
5.5  Write → feature_store/sparsity_flags.parquet

5.6  Compute log transforms on skewed columns:
     - out_sum_total_log, in_sum_total_log, ... (16 columns)
5.7  Write → feature_store/log_transforms.parquet
5.8  Release all intermediate DataFrames
```

**Memory**: ~80 MB  
**Classification**: SAFE

---

### Step 6: Staged Join → Training Table

```
6.1  Initialize empty DataFrame with integer index (850K rows)
6.2  FOR each of the 14 feature store parquet files:
     a) Read file (~5-25 MB each)
     b) Join to running DataFrame (integer index, no string keys)
     c) Release input file from RAM
     d) gc.collect()
6.3  fillna(0) on final DataFrame
6.4  Map integer index back to string ACCOUNT_ID
6.5  Split into train (595K) and test (255K) by ID mask
6.6  Merge train with churn labels
6.7  Write → processed_data/train_features.parquet
6.8  Write → processed_data/test_features.parquet
6.9  Release everything from RAM
```

**Memory**: ~800 MB peak (final 850K × 130 DataFrame + one small input file)  
**Classification**: SAFE

---

### Step 7: Model Training (with Leakage Fix)

```
7.1   Load train_features.parquet (~600 MB)
7.2   Load test_features.parquet (~250 MB)
7.3   Separate X, y, X_test (drop ACCOUNT_ID, CHURN)
7.4   Release raw DataFrames
7.5   Initialize StratifiedKFold(n_splits=10)

7.6   Train LightGBM (10 folds, balanced, early stopping)
7.7   gc.collect(); release non-essential model copies
7.8   Train XGBoost (10 folds, balanced, early stopping)
7.9   gc.collect()
7.10  Train CatBoost (if available)
7.11  gc.collect()
7.12  Train Random Forest (10 folds, balanced)
7.13  gc.collect()

7.14  FOR each fold in CV:
      a) Fit SimpleImputer on TRAIN FOLD ONLY
      b) Transform train fold, val fold, test set
      c) Fit StandardScaler on TRAIN FOLD ONLY
      d) Transform train fold, val fold, test set
      e) Train Logistic Regression on scaled train fold
      f) Train MLP on scaled train fold
      g) Predict on val fold and test set
      h) Release fold-specific scaled arrays
      i) gc.collect()

7.15  Save OOF predictions → predictions/oof_predictions.parquet
7.16  Save test predictions → predictions/test_predictions.parquet
7.17  Save fold-0 LGB model → models/lgb_model.pkl
7.18  Release all model objects except SHAP model
```

**Memory**: ~2 GB peak (feature matrix + largest model internal buffers)  
**Classification**: SAFE

---

### Step 8: Ensemble + Submission

```
8.1  Load oof_predictions.parquet (~50 MB)
8.2  Load test_predictions.parquet (~20 MB)
8.3  Compute individual base model AUCs
8.4  Train stacking meta-classifier (5-fold LR on OOF columns)
8.5  Compute Rank-Average Blend (AUC²-weighted)
8.6  Calibrate best method via LogisticRegression (Platt Scaling)
8.7  Sweep cost-sensitive threshold (5×FN + 1×FP)
8.8  Write → predictions.csv
```

**Memory**: ~100 MB  
**Classification**: SAFE

---

### Summary of File Read Count Reduction

| File | Current Reads | Rewritten Reads | Savings |
|------|--------------|----------------|---------|
| `trx_2024-01.parquet` | 3 (recency + march_adv scan + monthly agg) | 1 (single streaming pass) | 66% |
| `trx_2024-02.parquet` | 3 | 1 | 66% |
| `trx_2024-03.parquet` | 3 | 1 | 66% |
| `balance_2024-01.parquet` | 1 (but full load) | 1 (streaming) | RAM-safe |
| `balance_2024-02.parquet` | 1 (but full load) | 1 (streaming) | RAM-safe |
| `balance_2024-03.parquet` | 1 (but full load + pivot) | 1 (streaming + dense matrix) | RAM-safe |
| `kyc.parquet` | 1 | 1 | unchanged |
| **Total** | **13 full-file loads** | **7 streaming scans** | **46% fewer I/O ops** |

### Summary of Peak RAM Reduction

| Stage | Current Peak | Rewritten Peak |
|-------|-------------|---------------|
| Feature Engineering | ~12-15 GB | ~800 MB |
| Model Training | ~3.5 GB | ~2 GB |
| Ensemble | ~100 MB | ~100 MB |
| **Pipeline Maximum** | **~15 GB** | **~2 GB** |

---

> **End of memory-safe pipeline rewrite document.**
