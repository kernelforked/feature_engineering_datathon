# FictiPay Churn Prediction: Advanced Feature Catalog

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
