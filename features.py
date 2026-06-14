"""
features.py — Memory-Safe Streaming Feature Engineering Pipeline

Architecture:
  RAW DATA → STREAM PROCESSING (PyArrow iter_batches) → FEATURE STORE (parquet)
           → STAGED JOIN → TRAINING TABLE (compressed)

Key Design Principles:
  1. No full dataset is ever loaded into RAM
  2. All aggregations use numpy accumulators with np.add.at / np.maximum.at
  3. Each raw file is read exactly ONCE (single-pass multi-accumulator pattern)
  4. All intermediate results are persisted to disk in feature_store/
  5. Final join is staged (one file at a time) to avoid mega-join memory spikes
  6. ACCOUNT_ID is mapped to int64 for 12x memory reduction on joins

Peak RAM: ~800 MB (during staged join) vs ~15 GB (original pipeline)
"""

import os
import gc
import glob
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ─── Configuration ───────────────────────────────────────────────────────────
base_path = "/home/aspen/ProjectCollections/keggle/Datathon/bkash-presents-nsucec-datathon/public"

_script_dir = os.path.dirname(os.path.abspath(__file__))
FEATURE_STORE_DIR = os.path.join(_script_dir, "feature_store")
PROCESSED_DIR = os.path.join(_script_dir, "processed_data")

REF_DATE_RECENCY = pd.Timestamp("2024-03-31")           # used for global recency (matches original)
REF_DATE_MARCH   = pd.Timestamp("2024-03-31 23:59:59")  # used for March directional recency
REF_DATE_RECENCY_NS = REF_DATE_RECENCY.value
REF_DATE_MARCH_NS   = REF_DATE_MARCH.value
NS_PER_DAY = 86400 * 10**9

TRX_TYPES = ["P2P", "MerchantPay", "BillPay", "CashIn", "CashOut"]
TYPE_TO_INT = {t: i for i, t in enumerate(TRX_TYPES)}
OUTBOUND_TYPES = ["P2P", "MerchantPay", "BillPay", "CashOut"]

MONTHS = ["jan", "feb", "march"]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0: Initialization
# ═══════════════════════════════════════════════════════════════════════════════

def _init_dirs():
    """Create output directories."""
    os.makedirs(FEATURE_STORE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def build_id_mapping(sample_fraction=1.0, seed=42):
    """Load target IDs, optionally subsample, build string<->int mapping."""
    print("Loading target IDs...")
    train_labels = pd.read_csv(os.path.join(base_path, "train_labels.csv"))
    test_ids_df = pd.read_csv(os.path.join(base_path, "test.csv"))

    if sample_fraction < 1.0:
        print(f"  Sampling {sample_fraction*100}% of the data...")
        sampled = train_labels.groupby("CHURN", group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=seed)
        )
        train_sampled = sampled.reset_index(drop=True)
        test_sampled = test_ids_df.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)
    else:
        train_sampled = train_labels
        test_sampled = test_ids_df

    target_ids_list = sorted(set(train_sampled["ACCOUNT_ID"]) | set(test_sampled["ACCOUNT_ID"]))
    id_to_int = {aid: i for i, aid in enumerate(target_ids_list)}
    int_to_id = {i: aid for aid, i in id_to_int.items()}

    n = len(target_ids_list)
    print(f"  Total target customers: {n:,}")

    return target_ids_list, id_to_int, int_to_id, n, train_sampled, test_sampled


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: KYC Features
# ═══════════════════════════════════════════════════════════════════════════════

def stream_kyc_features(n, id_to_int):
    """Process KYC metadata and write to feature store.
    KYC file is small (~7 MB) so full load is safe."""
    print("\n[Step 1/6] Processing KYC features...")
    kyc = pd.read_parquet(os.path.join(base_path, "kyc.parquet"))

    # Map and filter to target IDs
    kyc["aid_int"] = kyc["ACCOUNT_ID"].map(id_to_int)
    kyc = kyc.dropna(subset=["aid_int"]).copy()
    kyc["aid_int"] = kyc["aid_int"].astype(int)

    # Compute tenure
    kyc["tenure_days"] = (REF_DATE_RECENCY - pd.to_datetime(kyc["ACCOUNT_OPEN_DATE"])).dt.days

    # Fill and encode demographics
    kyc["GENDER"] = kyc["GENDER"].fillna("Unknown")
    kyc["REGION"] = kyc["REGION"].fillna("Unknown")
    kyc = pd.get_dummies(kyc, columns=["GENDER", "REGION"], prefix=["gender", "region"], dtype=float)

    # Drop non-feature columns and set integer index
    drop_cols = ["ACCOUNT_ID", "ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE"]
    kyc = kyc.drop(columns=[c for c in drop_cols if c in kyc.columns])
    kyc = kyc.set_index("aid_int").sort_index()

    # Reindex to full range so all target IDs are present
    kyc = kyc.reindex(range(n), fill_value=0)

    kyc.to_parquet(os.path.join(FEATURE_STORE_DIR, "kyc_features.parquet"))
    print(f"  KYC features: {kyc.shape[1]} columns written.")
    del kyc
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Transaction Features (Single-Pass Per File)
# ═══════════════════════════════════════════════════════════════════════════════

def stream_transaction_features(n, id_to_int):
    """Single-pass-per-file transaction feature extraction using numpy accumulators.

    Each transaction file is read ONCE via PyArrow iter_batches.
    Multiple accumulators run simultaneously to compute:
      - Monthly outbound stats (count, sum, sum_sq, max)
      - Monthly inbound stats (count, sum)
      - Type-specific counts
      - Global recency (across all months)
      - March-specific: directional recency, type recency, micro-windows, velocity
    """
    print("\n[Step 2/6] Streaming transaction features (single-pass per file)...")

    trx_files = sorted(glob.glob(os.path.join(base_path, "transactions", "*.parquet")))
    if not trx_files:
        print("  WARNING: No transaction files found!")
        return

    # ─── Global recency accumulator (across all months, both directions) ───
    global_last_dt_ns = np.full(n, -1, dtype=np.int64)

    for month_idx, filepath in enumerate(trx_files):
        if month_idx >= len(MONTHS):
            break
        month = MONTHS[month_idx]
        is_march = (month == "march")

        print(f"\n  Processing {month} transactions ({os.path.basename(filepath)})...")

        # ─── Monthly accumulators (outbound / SRC side) ───
        out_count  = np.zeros(n, dtype=np.int64)
        out_sum    = np.zeros(n, dtype=np.float64)
        out_sum_sq = np.zeros(n, dtype=np.float64)
        out_max_val = np.full(n, -np.inf, dtype=np.float64)

        # ─── Monthly accumulators (inbound / DST side) ───
        in_count = np.zeros(n, dtype=np.int64)
        in_sum   = np.zeros(n, dtype=np.float64)

        # ─── Type counts (outbound, 5 types) ───
        type_counts = np.zeros((n, len(TRX_TYPES)), dtype=np.int64)

        # ─── March-specific accumulators ───
        if is_march:
            march_out_last_ns = np.full(n, -1, dtype=np.int64)
            march_in_last_ns  = np.full(n, -1, dtype=np.int64)

            march_type_last_ns = {t: np.full(n, -1, dtype=np.int64) for t in OUTBOUND_TYPES}
            march_dst_cashin_last_ns = np.full(n, -1, dtype=np.int64)
            march_dst_p2p_last_ns    = np.full(n, -1, dtype=np.int64)

            out_7d_count  = np.zeros(n, dtype=np.int64)
            out_7d_sum    = np.zeros(n, dtype=np.float64)
            out_14d_count = np.zeros(n, dtype=np.int64)
            out_14d_sum   = np.zeros(n, dtype=np.float64)
            in_7d_count   = np.zeros(n, dtype=np.int64)
            in_7d_sum     = np.zeros(n, dtype=np.float64)
            in_14d_count  = np.zeros(n, dtype=np.int64)
            in_14d_sum    = np.zeros(n, dtype=np.float64)

            MARCH_7D_NS  = pd.Timestamp("2024-03-25 00:00:00").value
            MARCH_14D_NS = pd.Timestamp("2024-03-18 00:00:00").value

        # ─── Stream through row groups ───
        columns = ["SRC_ACCOUNT", "DST_ACCOUNT", "TRX_TYPE", "TRX_AMT", "TRX_DATETIME"]
        pf = pq.ParquetFile(filepath)
        batch_num = 0

        for batch in pf.iter_batches(columns=columns):
            chunk = batch.to_pandas()
            batch_num += 1

            # Parse datetime as int64 nanoseconds for fast comparison
            dt_ns = pd.to_datetime(chunk["TRX_DATETIME"]).values.astype(np.int64)
            amt = chunk["TRX_AMT"].values.astype(np.float64)
            trx_type_ints = chunk["TRX_TYPE"].map(TYPE_TO_INT).values

            # ─── Source (Outbound) processing ───
            src_mapped = chunk["SRC_ACCOUNT"].map(id_to_int)
            src_valid_mask = src_mapped.notna().values
            if src_valid_mask.any():
                src_idx = src_mapped.values[src_valid_mask].astype(np.int64)
                src_amt = amt[src_valid_mask]
                src_dt  = dt_ns[src_valid_mask]
                src_types = trx_type_ints[src_valid_mask]

                # Basic aggregation
                np.add.at(out_count, src_idx, 1)
                np.add.at(out_sum, src_idx, src_amt)
                np.add.at(out_sum_sq, src_idx, src_amt ** 2)
                np.maximum.at(out_max_val, src_idx, src_amt)

                # Type counts (filter out any unexpected types that map to NaN)
                valid_type = ~np.isnan(src_types)
                if valid_type.any():
                    np.add.at(type_counts,
                              (src_idx[valid_type], src_types[valid_type].astype(np.int64)), 1)

                # Global recency (SRC side)
                np.maximum.at(global_last_dt_ns, src_idx, src_dt)

                if is_march:
                    np.maximum.at(march_out_last_ns, src_idx, src_dt)

                    # Type-specific outbound recency
                    for t in OUTBOUND_TYPES:
                        t_int = TYPE_TO_INT[t]
                        t_mask = src_types == t_int
                        if t_mask.any():
                            np.maximum.at(march_type_last_ns[t],
                                          src_idx[t_mask], src_dt[t_mask])

                    # Micro-windows (outbound)
                    mask_7d = src_dt >= MARCH_7D_NS
                    if mask_7d.any():
                        np.add.at(out_7d_count, src_idx[mask_7d], 1)
                        np.add.at(out_7d_sum, src_idx[mask_7d], src_amt[mask_7d])

                    mask_14d = src_dt >= MARCH_14D_NS
                    if mask_14d.any():
                        np.add.at(out_14d_count, src_idx[mask_14d], 1)
                        np.add.at(out_14d_sum, src_idx[mask_14d], src_amt[mask_14d])

            # ─── Destination (Inbound) processing ───
            dst_mapped = chunk["DST_ACCOUNT"].map(id_to_int)
            dst_valid_mask = dst_mapped.notna().values
            if dst_valid_mask.any():
                dst_idx = dst_mapped.values[dst_valid_mask].astype(np.int64)
                dst_amt = amt[dst_valid_mask]
                dst_dt  = dt_ns[dst_valid_mask]
                dst_types = trx_type_ints[dst_valid_mask]

                np.add.at(in_count, dst_idx, 1)
                np.add.at(in_sum, dst_idx, dst_amt)

                # Global recency (DST side)
                np.maximum.at(global_last_dt_ns, dst_idx, dst_dt)

                if is_march:
                    np.maximum.at(march_in_last_ns, dst_idx, dst_dt)

                    # CashIn received
                    cashin_mask = dst_types == TYPE_TO_INT["CashIn"]
                    if cashin_mask.any():
                        np.maximum.at(march_dst_cashin_last_ns,
                                      dst_idx[cashin_mask], dst_dt[cashin_mask])

                    # P2P received
                    p2p_mask = dst_types == TYPE_TO_INT["P2P"]
                    if p2p_mask.any():
                        np.maximum.at(march_dst_p2p_last_ns,
                                      dst_idx[p2p_mask], dst_dt[p2p_mask])

                    # Micro-windows (inbound)
                    mask_7d = dst_dt >= MARCH_7D_NS
                    if mask_7d.any():
                        np.add.at(in_7d_count, dst_idx[mask_7d], 1)
                        np.add.at(in_7d_sum, dst_idx[mask_7d], dst_amt[mask_7d])

                    mask_14d = dst_dt >= MARCH_14D_NS
                    if mask_14d.any():
                        np.add.at(in_14d_count, dst_idx[mask_14d], 1)
                        np.add.at(in_14d_sum, dst_idx[mask_14d], dst_amt[mask_14d])

            del chunk
            if batch_num % 10 == 0:
                gc.collect()

        # ─── Finalize monthly features ───
        print(f"    Finalizing {month} features ({batch_num} batches processed)...")

        monthly = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))

        monthly[f"out_count_{month}"] = out_count
        monthly[f"out_sum_{month}"]   = out_sum

        # Average: sum / count (0 for zero-count accounts)
        with np.errstate(divide='ignore', invalid='ignore'):
            out_avg = np.where(out_count > 0, out_sum / out_count, 0.0)
        monthly[f"out_avg_{month}"] = out_avg

        # Std dev: sqrt((n*sum_sq - sum^2) / (n*(n-1))) with Bessel correction
        with np.errstate(divide='ignore', invalid='ignore'):
            variance = np.where(
                out_count > 1,
                (out_sum_sq - (out_sum ** 2) / out_count) / (out_count - 1),
                0.0
            )
            variance = np.maximum(variance, 0.0)
        monthly[f"out_std_{month}"] = np.sqrt(variance)

        # Max (replace NaN with 0 for accounts with no transactions)
        monthly[f"out_max_{month}"] = np.where(np.isinf(out_max_val), 0.0, out_max_val)

        monthly[f"in_count_{month}"] = in_count
        monthly[f"in_sum_{month}"]   = in_sum

        # Type counts
        for i, ttype in enumerate(TRX_TYPES):
            monthly[f"count_{ttype}_{month}"] = type_counts[:, i]

        monthly.to_parquet(os.path.join(FEATURE_STORE_DIR, f"trx_agg_{month}.parquet"))
        print(f"    Wrote trx_agg_{month}.parquet ({monthly.shape[1]} columns)")

        # ─── March advanced features ───
        if is_march:
            print(f"    Finalizing March advanced features...")
            march_adv = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))

            # Directional recency (days since last outbound/inbound)
            march_adv["days_since_last_outbound"] = np.where(
                march_out_last_ns > 0,
                np.floor((REF_DATE_MARCH_NS - march_out_last_ns) / NS_PER_DAY),
                31.0
            )
            march_adv["days_since_last_inbound"] = np.where(
                march_in_last_ns > 0,
                np.floor((REF_DATE_MARCH_NS - march_in_last_ns) / NS_PER_DAY),
                31.0
            )

            # Type-specific outbound recency
            for t in OUTBOUND_TYPES:
                march_adv[f"days_since_last_{t}"] = np.where(
                    march_type_last_ns[t] > 0,
                    np.floor((REF_DATE_MARCH_NS - march_type_last_ns[t]) / NS_PER_DAY),
                    31.0
                )

            # Inbound type recency
            march_adv["days_since_last_CashIn"] = np.where(
                march_dst_cashin_last_ns > 0,
                np.floor((REF_DATE_MARCH_NS - march_dst_cashin_last_ns) / NS_PER_DAY),
                31.0
            )
            march_adv["days_since_received_P2P"] = np.where(
                march_dst_p2p_last_ns > 0,
                np.floor((REF_DATE_MARCH_NS - march_dst_p2p_last_ns) / NS_PER_DAY),
                31.0
            )

            # Micro-window counts and sums
            march_adv["out_count_last_7d"]  = out_7d_count
            march_adv["out_sum_last_7d"]    = out_7d_sum
            march_adv["out_count_last_14d"] = out_14d_count
            march_adv["out_sum_last_14d"]   = out_14d_sum
            march_adv["in_count_last_7d"]   = in_7d_count
            march_adv["in_sum_last_7d"]     = in_7d_sum
            march_adv["in_count_last_14d"]  = in_14d_count
            march_adv["in_sum_last_14d"]    = in_14d_sum

            # Velocity ratios (micro-window / total March)
            march_out_ct  = out_count.astype(np.float64)
            march_out_sm  = out_sum.copy()
            march_adv["out_count_velocity_7d"]  = out_7d_count  / (march_out_ct + 1e-5)
            march_adv["out_count_velocity_14d"] = out_14d_count / (march_out_ct + 1e-5)
            march_adv["out_sum_velocity_7d"]    = out_7d_sum    / (march_out_sm + 1e-5)
            march_adv["out_sum_velocity_14d"]   = out_14d_sum   / (march_out_sm + 1e-5)

            march_adv.to_parquet(os.path.join(FEATURE_STORE_DIR, "trx_march_advanced.parquet"))
            print(f"    Wrote trx_march_advanced.parquet ({march_adv.shape[1]} columns)")

            del march_adv, march_out_last_ns, march_in_last_ns
            del march_type_last_ns, march_dst_cashin_last_ns, march_dst_p2p_last_ns
            del out_7d_count, out_7d_sum, out_14d_count, out_14d_sum
            del in_7d_count, in_7d_sum, in_14d_count, in_14d_sum

        del monthly, out_count, out_sum, out_sum_sq, out_max_val
        del in_count, in_sum, type_counts
        gc.collect()

    # ─── Global recency (across all 3 months) ───
    print("\n  Finalizing global recency...")
    recency_days = np.where(
        global_last_dt_ns > 0,
        np.floor((REF_DATE_RECENCY_NS - global_last_dt_ns) / NS_PER_DAY),
        90.0
    )
    recency_df = pd.DataFrame(
        {"days_since_last_trx": recency_days},
        index=pd.RangeIndex(n, name="aid_int")
    )
    recency_df.to_parquet(os.path.join(FEATURE_STORE_DIR, "trx_recency.parquet"))
    print(f"  Wrote trx_recency.parquet")

    del global_last_dt_ns, recency_df
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Balance Features (Streaming)
# ═══════════════════════════════════════════════════════════════════════════════

def stream_balance_features(n, id_to_int):
    """Streaming balance feature extraction using PyArrow iter_batches.

    For monthly stats: online Welford accumulators for mean/std/min/max.
    For March daily features: scatter into dense numpy matrix (850K × 31 float32 ≈ 105 MB).
    """
    print("\n[Step 3/6] Streaming balance features...")

    bal_files = sorted(glob.glob(os.path.join(base_path, "dayend_balance", "*.parquet")))

    for month_idx, filepath in enumerate(bal_files):
        if month_idx >= len(MONTHS):
            break
        month = MONTHS[month_idx]
        is_march = (month == "march")

        print(f"\n  Processing {month} balances ({os.path.basename(filepath)})...")

        # Online accumulators for mean/std/min/max
        bal_count  = np.zeros(n, dtype=np.int64)
        bal_sum    = np.zeros(n, dtype=np.float64)
        bal_sum_sq = np.zeros(n, dtype=np.float64)
        bal_min_val = np.full(n, np.inf, dtype=np.float64)
        bal_max_val = np.full(n, -np.inf, dtype=np.float64)

        # March daily matrix (n × 31, float32 ≈ 105 MB for 850K accounts)
        if is_march:
            daily_bal = np.zeros((n, 31), dtype=np.float32)

        pf = pq.ParquetFile(filepath)
        columns = ["ACCOUNT_ID", "AVAILABLE_BALANCE"]
        if is_march:
            columns.append("DATE")

        batch_num = 0
        for batch in pf.iter_batches(columns=columns):
            chunk = batch.to_pandas()
            batch_num += 1

            # Map and filter to target IDs
            mapped = chunk["ACCOUNT_ID"].map(id_to_int)
            valid_mask = mapped.notna().values
            if not valid_mask.any():
                del chunk
                continue

            idx = mapped.values[valid_mask].astype(np.int64)
            bal = chunk["AVAILABLE_BALANCE"].values[valid_mask].astype(np.float64)

            # Accumulate stats
            np.add.at(bal_count, idx, 1)
            np.add.at(bal_sum, idx, bal)
            np.add.at(bal_sum_sq, idx, bal ** 2)
            np.minimum.at(bal_min_val, idx, bal)
            np.maximum.at(bal_max_val, idx, bal)

            # March daily scatter
            if is_march:
                day_idx = pd.to_datetime(chunk["DATE"].values[valid_mask]).day - 1  # 0-indexed
                daily_bal[idx, day_idx] = bal.astype(np.float32)

            del chunk
            if batch_num % 20 == 0:
                gc.collect()

        # ─── Finalize monthly stats ───
        print(f"    Finalizing {month} balance stats ({batch_num} batches)...")
        bal_monthly = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_bal = np.where(bal_count > 0, bal_sum / bal_count, 0.0)
            variance = np.where(
                bal_count > 1,
                (bal_sum_sq - (bal_sum ** 2) / bal_count) / (bal_count - 1),
                0.0
            )
            variance = np.maximum(variance, 0.0)

        bal_monthly[f"mean_bal_{month}"] = mean_bal
        bal_monthly[f"std_bal_{month}"]  = np.sqrt(variance)
        bal_monthly[f"min_bal_{month}"]  = np.where(np.isinf(bal_min_val), 0.0, bal_min_val)
        bal_monthly[f"max_bal_{month}"]  = np.where(np.isinf(bal_max_val), 0.0, bal_max_val)

        bal_monthly.to_parquet(os.path.join(FEATURE_STORE_DIR, f"bal_agg_{month}.parquet"))
        print(f"    Wrote bal_agg_{month}.parquet")

        # ─── March advanced features from dense daily matrix ───
        if is_march:
            print(f"    Computing March daily balance features...")
            march_bal = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))

            daily_f64 = daily_bal.astype(np.float64)

            # Final balance (day 31 = index 30)
            march_bal["final_balance_march"] = daily_f64[:, 30]

            # Balance drop (day 31 - day 1)
            march_bal["balance_drop_march"] = daily_f64[:, 30] - daily_f64[:, 0]

            # Linear trend slope (31 days), centered at day 16
            t_centered = np.arange(31, dtype=np.float64) - 15.0
            denom = (t_centered ** 2).sum()
            weights = t_centered / denom
            march_bal["balance_trend_march"] = daily_f64 @ weights

            # Zero balance days (< 10 TK)
            march_bal["zero_balance_days_march"] = (daily_bal < 10.0).sum(axis=1).astype(int)

            # Mean balance last 7 days (days 25-31 = indices 24-30)
            march_bal["mean_balance_last_7d_march"] = daily_f64[:, 24:31].mean(axis=1)

            # Mean balance last 14 days (days 18-31 = indices 17-30)
            march_bal["mean_balance_last_14d_march"] = daily_f64[:, 17:31].mean(axis=1)

            # Zero balance days last 7 days
            march_bal["zero_balance_days_last_7d_march"] = (daily_bal[:, 24:31] < 10.0).sum(axis=1).astype(int)

            # Linear trend last 14 days
            t_14 = np.arange(14, dtype=np.float64) - 6.5
            denom_14 = (t_14 ** 2).sum()
            weights_14 = t_14 / denom_14
            march_bal["balance_trend_last_14d_march"] = daily_f64[:, 17:31] @ weights_14

            march_bal.to_parquet(os.path.join(FEATURE_STORE_DIR, "bal_march_advanced.parquet"))
            print(f"    Wrote bal_march_advanced.parquet ({march_bal.shape[1]} columns)")

            del daily_bal, daily_f64, march_bal

        del bal_monthly, bal_count, bal_sum, bal_sum_sq, bal_min_val, bal_max_val
        gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Cross-Month & Balance-Derived Features
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cross_month_features():
    """Compute derived cross-month features from the feature store.
    Loads only the small aggregated parquet files (~20-50 MB each)."""
    print("\n[Step 4/6] Computing cross-month derived features...")

    # Load monthly transaction aggregations
    trx_dfs = {}
    for month in MONTHS:
        fpath = os.path.join(FEATURE_STORE_DIR, f"trx_agg_{month}.parquet")
        if os.path.exists(fpath):
            trx_dfs[month] = pd.read_parquet(fpath)

    n = len(trx_dfs[MONTHS[0]])
    cross = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))

    # ─── Transaction totals ───
    cross["out_count_total"] = sum(trx_dfs[m][f"out_count_{m}"] for m in trx_dfs)
    cross["out_sum_total"]   = sum(trx_dfs[m][f"out_sum_{m}"] for m in trx_dfs)
    cross["in_count_total"]  = sum(trx_dfs[m][f"in_count_{m}"] for m in trx_dfs)
    cross["in_sum_total"]    = sum(trx_dfs[m][f"in_sum_{m}"] for m in trx_dfs)

    # ─── Decay features ───
    if "jan" in trx_dfs and "feb" in trx_dfs:
        cross["trx_count_decay_jan_feb"] = (
            (trx_dfs["feb"]["out_count_feb"] - trx_dfs["jan"]["out_count_jan"])
            / (trx_dfs["jan"]["out_count_jan"] + 1)
        )
    if "feb" in trx_dfs and "march" in trx_dfs:
        cross["trx_count_decay_feb_march"] = (
            (trx_dfs["march"]["out_count_march"] - trx_dfs["feb"]["out_count_feb"])
            / (trx_dfs["feb"]["out_count_feb"] + 1)
        )
    if "jan" in trx_dfs and "march" in trx_dfs:
        cross["trx_count_decay_jan_march"] = (
            (trx_dfs["march"]["out_count_march"] - trx_dfs["jan"]["out_count_jan"])
            / (trx_dfs["jan"]["out_count_jan"] + 1)
        )

    # ─── March activity share ───
    if "march" in trx_dfs:
        cross["march_activity_share"] = (
            trx_dfs["march"]["out_count_march"] / (cross["out_count_total"] + 1e-5)
        )

    # ─── Type totals ───
    for ttype in TRX_TYPES:
        cross[f"count_{ttype}_total"] = sum(
            trx_dfs[m].get(f"count_{ttype}_{m}", 0) for m in trx_dfs
        )

    # ─── Service diversity ───
    type_total_cols = [f"count_{t}_total" for t in TRX_TYPES]
    cross["trx_type_diversity"] = (cross[type_total_cols] > 0).sum(axis=1).astype(int)

    # ─── Ratios ───
    denom = cross["out_count_total"] + 1e-5
    cross["merchant_pay_ratio"] = cross["count_MerchantPay_total"] / denom
    cross["bill_pay_ratio"]     = cross["count_BillPay_total"] / denom
    cross["p2p_ratio"]          = cross["count_P2P_total"] / denom
    cross["cashout_ratio"]      = cross["count_CashOut_total"] / denom

    # ─── Net flow March ───
    if "march" in trx_dfs:
        cross["net_flow_march"] = (
            trx_dfs["march"]["in_sum_march"] - trx_dfs["march"]["out_sum_march"]
        )

    cross.to_parquet(os.path.join(FEATURE_STORE_DIR, "trx_cross_month.parquet"))
    print(f"  Wrote trx_cross_month.parquet ({cross.shape[1]} columns)")

    del trx_dfs, cross
    gc.collect()

    # ─── Balance-derived features ───
    print("  Computing balance-derived features...")
    bal_jan = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "bal_agg_jan.parquet"))
    bal_feb = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "bal_agg_feb.parquet"))
    bal_mar = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "bal_agg_march.parquet"))
    bal_mar_adv = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "bal_march_advanced.parquet"))

    bal_derived = pd.DataFrame(index=pd.RangeIndex(len(bal_mar), name="aid_int"))

    bal_derived["balance_stability_march"] = (
        bal_mar["std_bal_march"] / (bal_mar["mean_bal_march"] + 1e-5)
    )
    bal_derived["balance_change_jan_march"] = (
        bal_mar["mean_bal_march"] - bal_jan["mean_bal_jan"]
    )
    bal_derived["balance_change_feb_march"] = (
        bal_mar["mean_bal_march"] - bal_feb["mean_bal_feb"]
    )
    bal_derived["final_to_mean_balance_ratio_march"] = (
        bal_mar_adv["final_balance_march"] / (bal_mar["mean_bal_march"] + 1e-5)
    )

    bal_derived.to_parquet(os.path.join(FEATURE_STORE_DIR, "bal_derived.parquet"))
    print(f"  Wrote bal_derived.parquet ({bal_derived.shape[1]} columns)")

    del bal_jan, bal_feb, bal_mar, bal_mar_adv, bal_derived
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Abstract Features, Sparsity Flags, Log Transforms
# ═══════════════════════════════════════════════════════════════════════════════

def compute_abstract_and_derived():
    """Compute abstract features, sparsity flags, and log transforms from feature store."""
    print("\n[Step 5/6] Computing abstract features, flags, and log transforms...")

    # Load required feature store files (each < 30 MB)
    cross      = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "trx_cross_month.parquet"))
    recency    = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "trx_recency.parquet"))
    bal_march  = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "bal_agg_march.parquet"))
    bal_mar_adv = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "bal_march_advanced.parquet"))
    march_adv  = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "trx_march_advanced.parquet"))
    march_trx  = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "trx_agg_march.parquet"))
    bal_derived = pd.read_parquet(os.path.join(FEATURE_STORE_DIR, "bal_derived.parquet"))

    n = len(cross)

    # ─── Abstract Features ───
    print("  Computing abstract features...")
    abstract = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))

    # 1. Recency Pressure Index
    ati = 90.0 / (cross["out_count_total"] + 1)
    abstract["recency_pressure"] = recency["days_since_last_trx"] / (ati + 1e-5)

    # 2. Digital Wallet Integration Score
    abstract["digital_integration"] = cross["trx_type_diversity"] * (
        cross["bill_pay_ratio"] + cross["merchant_pay_ratio"]
        + cross["p2p_ratio"] - cross["cashout_ratio"]
    )

    # 3. Financial Depletion Velocity
    abstract["depletion_velocity"] = cross["trx_count_decay_feb_march"] + (
        bal_mar_adv["balance_drop_march"] / (bal_march["mean_bal_march"] + 1e-5)
    )

    # 4. Store of Value Index (Wallet Trust)
    abstract["store_of_value"] = (
        bal_march["mean_bal_march"] / (cross["in_sum_total"] + 1)
    ) * (1.0 - bal_derived["balance_stability_march"])

    # 5. Peer-to-Peer Network Stickiness
    abstract["network_stickiness"] = np.log1p(cross["count_P2P_total"]) * (
        31.0 / (march_adv["days_since_last_P2P"]
                + march_adv["days_since_received_P2P"] + 1.0)
    )

    # 6. Wallet Strain Index
    abstract["wallet_strain"] = (
        march_trx["out_std_march"] / (bal_march["mean_bal_march"] + 1e-5)
    )

    abstract.to_parquet(os.path.join(FEATURE_STORE_DIR, "abstract_features.parquet"))
    print(f"  Wrote abstract_features.parquet ({abstract.shape[1]} columns)")

    # ─── Sparsity Flags ───
    print("  Computing sparsity flags...")
    flags = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))

    flags["flag_zero_bill_pay"]     = (cross["bill_pay_ratio"] == 0).astype(float)
    flags["flag_zero_merchant_pay"] = (cross["merchant_pay_ratio"] == 0).astype(float)
    flags["flag_zero_march_trx"]    = (march_trx["out_count_march"] == 0).astype(float)
    flags["flag_zero_trx_last_7d_march"] = (march_adv["out_count_last_7d"] == 0).astype(float)

    flags.to_parquet(os.path.join(FEATURE_STORE_DIR, "sparsity_flags.parquet"))
    print(f"  Wrote sparsity_flags.parquet ({flags.shape[1]} columns)")

    # ─── Log Transforms ───
    print("  Computing log transforms...")
    log_sources = {
        "out_sum_total":  cross["out_sum_total"],
        "in_sum_total":   cross["in_sum_total"],
        "out_sum_march":  march_trx["out_sum_march"],
        "in_sum_march":   march_trx["in_sum_march"],
        "mean_bal_march": bal_march["mean_bal_march"],
        "std_bal_march":  bal_march["std_bal_march"],
        "final_balance_march":        bal_mar_adv["final_balance_march"],
        "out_sum_last_7d":            march_adv["out_sum_last_7d"],
        "out_sum_last_14d":           march_adv["out_sum_last_14d"],
        "in_sum_last_7d":             march_adv["in_sum_last_7d"],
        "in_sum_last_14d":            march_adv["in_sum_last_14d"],
        "mean_balance_last_7d_march": bal_mar_adv["mean_balance_last_7d_march"],
        "mean_balance_last_14d_march": bal_mar_adv["mean_balance_last_14d_march"],
        "network_stickiness":         abstract["network_stickiness"],
        "wallet_strain":              abstract["wallet_strain"],
    }

    log_df = pd.DataFrame(index=pd.RangeIndex(n, name="aid_int"))
    for col, series in log_sources.items():
        log_df[f"{col}_log"] = np.log1p(series.clip(lower=0))

    log_df.to_parquet(os.path.join(FEATURE_STORE_DIR, "log_transforms.parquet"))
    print(f"  Wrote log_transforms.parquet ({log_df.shape[1]} columns)")

    del cross, recency, bal_march, bal_mar_adv, march_adv, march_trx
    del bal_derived, abstract, flags, log_df
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Staged Join → Training Table
# ═══════════════════════════════════════════════════════════════════════════════

def staged_join_and_split(n, int_to_id, train_sampled, test_sampled):
    """Sequential join of feature store files, then split into train/test."""
    print("\n[Step 6/6] Staged join and train/test split...")

    feature_store_files = [
        "kyc_features.parquet",
        "trx_recency.parquet",
        "trx_agg_jan.parquet",
        "trx_agg_feb.parquet",
        "trx_agg_march.parquet",
        "trx_march_advanced.parquet",
        "trx_cross_month.parquet",
        "bal_agg_jan.parquet",
        "bal_agg_feb.parquet",
        "bal_agg_march.parquet",
        "bal_march_advanced.parquet",
        "bal_derived.parquet",
        "abstract_features.parquet",
        "sparsity_flags.parquet",
        "log_transforms.parquet",
    ]

    # Build final table by sequential column-merge (all share same integer index)
    final = None
    for fname in feature_store_files:
        fpath = os.path.join(FEATURE_STORE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found, skipping.")
            continue

        part = pd.read_parquet(fpath)
        if final is None:
            final = part
        else:
            final = final.join(part, how="left")
            del part
        gc.collect()
        print(f"  Joined {fname} → {final.shape[1]} total columns")

    final = final.fillna(0)

    # Map integer index back to ACCOUNT_ID
    final["ACCOUNT_ID"] = final.index.map(int_to_id)
    final = final.reset_index(drop=True)

    # Split into train and test
    train_ids_set = set(train_sampled["ACCOUNT_ID"])
    test_ids_set  = set(test_sampled["ACCOUNT_ID"])

    train_mask = final["ACCOUNT_ID"].isin(train_ids_set)
    test_mask  = final["ACCOUNT_ID"].isin(test_ids_set)

    train_features = final[train_mask].copy()
    train_features = train_features.merge(
        train_sampled[["ACCOUNT_ID", "CHURN"]], on="ACCOUNT_ID", how="left"
    )

    test_features = final[test_mask].copy()

    del final
    gc.collect()

    print(f"\n  Train features: {train_features.shape}")
    print(f"  Test features: {test_features.shape}")

    train_features.to_parquet(os.path.join(PROCESSED_DIR, "train_features.parquet"))
    test_features.to_parquet(os.path.join(PROCESSED_DIR, "test_features.parquet"))
    print("  Features saved to processed_data/.")

    del train_features, test_features
    gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(sample_fraction=1.0, seed=42):
    """Run end-to-end memory-safe feature engineering pipeline."""
    print(f"\n{'='*60}")
    print(f"  Memory-Safe Feature Engineering Pipeline")
    print(f"  Sample: {sample_fraction*100}%, Seed: {seed}")
    print(f"{'='*60}")

    _init_dirs()

    # Step 0: Build ID mapping
    target_ids_list, id_to_int, int_to_id, n, train_sampled, test_sampled = \
        build_id_mapping(sample_fraction, seed)

    # Step 1: KYC features
    stream_kyc_features(n, id_to_int)

    # Step 2: Transaction features (single pass per file)
    stream_transaction_features(n, id_to_int)

    # Step 3: Balance features (streaming)
    stream_balance_features(n, id_to_int)

    # Step 4: Cross-month & balance-derived features
    compute_cross_month_features()

    # Step 5: Abstract features, flags, log transforms
    compute_abstract_and_derived()

    # Step 6: Staged join and split
    staged_join_and_split(n, int_to_id, train_sampled, test_sampled)

    # Write feature catalog
    generate_feature_catalog()

    print(f"\n{'='*60}")
    print(f"  Feature Engineering Complete!")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Catalog
# ═══════════════════════════════════════════════════════════════════════════════

def generate_feature_catalog():
    """Write the features.md catalog."""
    catalog = """# FictiPay Churn Prediction: Advanced Feature Catalog

## 1. Tenure & Demographics (KYC)
- **tenure_days**: Days from ACCOUNT_OPEN_DATE to 2024-03-31.
- **gender_*, region_***: One-hot encoded demographics.

## 2. General Transaction Activity (Jan + Feb + March)
- **out_count_[jan/feb/march/total]**: Outbound transaction counts per month and total.
- **out_sum_[jan/feb/march/total]**: Total outbound amount (TK) per month.
- **out_avg_[jan/feb/march]**: Average transaction size per month.
- **out_std_[jan/feb/march]**: Std dev of transaction amounts.
- **out_max_[jan/feb/march]**: Max single transaction.
- **in_count_[jan/feb/march/total]**: Inbound transaction counts.
- **in_sum_[jan/feb/march/total]**: Total inbound volume.

## 3. Directional & Type Recency (March Advanced)
- **days_since_last_trx**: Days between last transaction (any direction) and 2024-03-31.
- **days_since_last_outbound**: Days since last outbound transaction.
- **days_since_last_inbound**: Days since last inbound transaction.
- **days_since_last_P2P / days_since_last_MerchantPay / days_since_last_BillPay / days_since_last_CashOut**: Outbound recency per type.
- **days_since_last_CashIn / days_since_received_P2P**: Inbound recency per type.

## 4. Micro-Windows & Velocity (March Advanced)
- **out_count_last_7d / out_sum_last_7d**: Count and sum of outbound transactions in the last week of March (March 25-31).
- **out_count_last_14d / out_sum_last_14d**: Count and sum of outbound transactions in the last 2 weeks of March (March 18-31).
- **in_count_last_7d / in_sum_last_7d / in_count_last_14d / in_sum_last_14d**: Inbound micro-window stats.
- **out_count_velocity_7d / out_count_velocity_14d / out_sum_velocity_7d / out_sum_velocity_14d**: Ratio of micro-window spend/count to the total March spend/count.

## 5. Temporal Decay (3-Month Trends)
- **trx_count_decay_jan_feb / trx_count_decay_feb_march / trx_count_decay_jan_march**: Rate of activity change.
- **march_activity_share**: March count / total count.

## 6. Service Diversity & Type Ratios
- **count_[P2P/MerchantPay/BillPay/CashIn/CashOut]_total**: Type-level total activity.
- **trx_type_diversity**: Number of distinct types used (0-5).
- **merchant_pay_ratio / bill_pay_ratio / p2p_ratio / cashout_ratio**: Spend composition.
- **flag_zero_bill_pay / flag_zero_merchant_pay / flag_zero_march_trx / flag_zero_trx_last_7d_march**: Zero-inflation/zero-activity flags.

## 7. Net Flow
- **net_flow_march**: March inbound minus outbound. Negative = wallet draining.

## 8. Balance Trends & Micro-Windows
- **mean_bal_[jan/feb/march]**: Average daily balance per month.
- **mean_balance_last_7d_march / mean_balance_last_14d_march**: Average balance in the last week and two weeks of March.
- **final_balance_march**: Balance on March 31.
- **balance_drop_march**: March 31 - March 1 change.
- **balance_trend_march**: Linear slope of daily balance over March (31 days).
- **balance_trend_last_14d_march**: Linear slope of daily balance over last 14 days.
- **zero_balance_days_march / zero_balance_days_last_7d_march**: Days with balance < 10 TK.
- **balance_stability_march**: CV (std/mean) of March balance.
- **balance_change_jan_march / balance_change_feb_march**: Cross-month balance trends.
- **final_to_mean_balance_ratio_march**: Final balance relative to mean balance.

## 9. Domain-Specific Custom Abstractions
- **recency_pressure**: Normalized dormancy index (days since last transaction divided by customer average transaction interval).
- **digital_integration**: Measure of digital wallet utility (diversity of transactions scaled by digital spend ratio vs cash-out dependency).
- **depletion_velocity**: Rate of wallet emptying and transactional slowdown.
- **store_of_value**: Wallet trust index (mean balance relative to total inbound cash-in, scaled by balance stability).
- **network_stickiness**: P2P social network anchoring value.
- **wallet_strain**: Index of transaction size volatility relative to the customer's average wallet balance.
"""
    catalog_path = os.path.join(_script_dir, "features.md")
    with open(catalog_path, "w") as f:
        f.write(catalog)
    print("Feature catalog features.md updated.")


if __name__ == "__main__":
    build_features(sample_fraction=1.0)
