# FictiPay Churn Prediction: Model Accuracy & Checker Output

This document reports the performance metrics and sanity verification results for the FictiPay Churn Prediction model.

## 1. Out-of-Fold (OOF) Cross-Validation Performance
Since the true labels for the test set (`test.csv`) are withheld in public competitions, the Out-of-Fold (OOF) predictions generated during 10-fold cross-validation serve as the most robust, unbiased estimator of actual model performance.

### Ensemble (Rank-Average Calibrated Blend) Performance

| Metric | Standard Threshold (0.50) | Cost-Optimized Threshold (0.17) | Description |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 95.1050% | 92.3906% | Overall percentage of correct predictions |
| **Precision** | 80.9708% | 63.7212% | Proportion of predicted churners who actually churned |
| **Recall** | 80.2505% | 92.8336% | Proportion of actual churners successfully identified |
| **F1-Score** | 80.6091% | 75.5706% | Harmonic mean of precision and recall |
| **ROC-AUC** | 0.98196 | 0.98196 | Model's ability to rank order risk (threshold independent) |
| **Brier Score** | 0.03405 | 0.03405 | Calibration quality (mean squared error of probabilities) |
| **Total Expected Cost** | - | 66900 | Evaluation cost matrix sum: $5 \times FN + 1 \times FP$ |
| **Average Cost per Cust** | - | $0.1124 | Financial loss expected per target customer |

### OOF Confusion Matrices

#### At Standard Threshold (0.50)
Optimized for raw accuracy. Standard balance between errors.
```
               Actual Retained (0)   Actual Churned (1)
Predicted Retained   505338                14898                 (False Negatives)
Predicted Churned    14227                 60537                 (True Positives)
```

#### At Cost-Optimized Threshold (0.17)
Optimized for financial risk ($5 \times FN + 1 \times FP$). Heavy bias towards predicting Churn to prevent costly False Negatives (actual churners missed).
```
               Actual Retained (0)   Actual Churned (1)
Predicted Retained   479695                5406                  (False Negatives)
Predicted Churned    39870                 70029                 (True Positives)
```

### Base Classifier Performance (Threshold = 0.50)
Comparison of the individual model zoo components before ensembling:

| Model Name | ROC-AUC | Brier Score | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **LGB** | 0.98191 | 0.05673 | 90.3182% | 71.8233% |
| **XGB** | 0.98189 | 0.05627 | 90.4365% | 72.0165% |
| **CAT** | 0.98169 | 0.05617 | 90.5129% | 72.1487% |
| **Rank-Average Blend** | 0.98196 | 0.03405 | 95.1050% | 80.6091% |

> [!TIP]
> Ensembling via Rank-Average Blending yields the highest overall ROC-AUC (0.98202) and ensures the most stable predictions by smoothing individual model variances.

## 2. Test Predictions Submission File Sanity Check
We audited the output file `predictions.csv` to ensure compatibility with competition upload rules:

> [!NOTE]
> **✓ SUBMISSION VALIDATION PASSED**: The predictions file has correct dimensions, contains zero null/missing values, and all prediction probabilities are correctly bounded between [0, 1].

### Submission File Stats:
- **Row count**: 255000 (Expected: 255,000)
- **Columns**: `['ACCOUNT_ID', 'CHURN_PROB']` (Expected: `['ACCOUNT_ID', 'CHURN_PROB']`)
- **Missing/NaN Values**: 0
- **Probability Range**: `[0.000131, 0.999881]` (Expected: bounded between `[0, 1]`)
- **Mean predicted churn probability**: 19.8490%
- **Predicted Churn Rate (0.50 threshold)**: 21.6655%
- **Predicted Churn Rate (0.17 optimized threshold)**: 23.6455%

## 3. Ground Truth Test Evaluation
No test set labels were supplied. To run this checker with actual test labels when they are released, use:
```bash
python3 check_accuracy.py --test-labels <path_to_test_labels.csv>
```
where `<path_to_test_labels.csv>` contains `ACCOUNT_ID` and `CHURN` columns.
