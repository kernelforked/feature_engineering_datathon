"""
ensemble.py — Stacking Meta-Learner, Rank-Average Blending, and Probability Calibration

This module loads the out-of-fold (OOF) predictions from each base model,
compares Stacking meta-models and Rank-Average Blending, calibrates the final output
via Isotonic Regression, and generates the submission CSV.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from scipy.stats import rankdata

def load_predictions():
    """Load OOF and test predictions from the model zoo."""
    oof = pd.read_parquet("./predictions/oof_predictions.parquet")
    test = pd.read_parquet("./predictions/test_predictions.parquet")
    return oof, test

def build_stacking_ensemble():
    """Train a Logistic Regression/Ridge meta-classifier on stacked OOF predictions and compare with Rank-Average Blending."""
    oof, test = load_predictions()
    
    # Identify available base models dynamically
    stack_cols = [c for c in oof.columns if c.endswith("_oof")]
    test_stack_cols = [c.replace("_oof", "_test") for c in stack_cols]
    
    print("Detected base models for ensembling:")
    for col in stack_cols:
        print(f"  - {col[:-4]}")
        
    X_stack = oof[stack_cols].values
    y = oof["CHURN"].values
    X_test_stack = test[test_stack_cols].values
    
    # Results dictionary
    results = {}
    
    # 1. Individual Model Metrics
    print("\n--- Individual Base Model AUCs ---")
    individual_aucs = {}
    for col in stack_cols:
        auc = roc_auc_score(y, oof[col].values)
        brier = brier_score_loss(y, oof[col].values)
        individual_aucs[col] = auc
        print(f"  {col[:-4]:<20}: AUC = {auc:.5f}, Brier = {brier:.5f}")
        results[f"Base: {col[:-4]}"] = {
            "auc": auc,
            "brier": brier,
            "oof": oof[col].values,
            "test": test[col.replace('_oof', '_test')].values
        }

    import lightgbm as lgb
    
    # 2. Stacking Meta-Classifier (LightGBM)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(X_stack))
    meta_test = np.zeros(len(X_test_stack))
    
    print("\nTraining Stacking Meta-Classifier (LightGBM)...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_stack, y)):
        X_train, y_train = X_stack[train_idx], y[train_idx]
        X_val, y_val = X_stack[val_idx], y[val_idx]
        
        meta_model = lgb.LGBMClassifier(
            n_estimators=200, 
            learning_rate=0.01, 
            max_depth=3, 
            num_leaves=7, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42 + fold, 
            n_jobs=-1,
            verbose=-1
        )
        meta_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        meta_oof[val_idx] = meta_model.predict_proba(X_val)[:, 1]
        meta_test += meta_model.predict_proba(X_test_stack)[:, 1] / cv.n_splits
    
    results["Stacking (LightGBM)"] = {
        "auc": roc_auc_score(y, meta_oof),
        "brier": brier_score_loss(y, meta_oof),
        "oof": meta_oof,
        "test": meta_test
    }
    
    # 3. Simple Weighted Average Ensemble
    total_auc = sum(individual_aucs.values())
    weights = {col: auc / total_auc for col, auc in individual_aucs.items()}
    
    weighted_oof = np.zeros(len(X_stack))
    weighted_test = np.zeros(len(X_test_stack))
    for oof_col, test_col in zip(stack_cols, test_stack_cols):
        w = weights[oof_col]
        weighted_oof += w * oof[oof_col].values
        weighted_test += w * test[test_col].values
        
    results["Weighted Average"] = {
        "auc": roc_auc_score(y, weighted_oof),
        "brier": brier_score_loss(y, weighted_oof),
        "oof": weighted_oof,
        "test": weighted_test
    }
    
    # Calibrate Weighted Average using Logistic Regression (Platt Scaling) to preserve ranking order
    lr_cal_w = LogisticRegression(random_state=42)
    lr_cal_w.fit(weighted_oof.reshape(-1, 1), y)
    calibrated_weighted_oof = lr_cal_w.predict_proba(weighted_oof.reshape(-1, 1))[:, 1]
    calibrated_weighted_test = lr_cal_w.predict_proba(weighted_test.reshape(-1, 1))[:, 1]
    
    results["Weighted Average (Logistic Calibrated)"] = {
        "auc": roc_auc_score(y, calibrated_weighted_oof),
        "brier": brier_score_loss(y, calibrated_weighted_oof),
        "oof": calibrated_weighted_oof,
        "test": calibrated_weighted_test
    }

    # 4. Rank-Average Blending (Kaggle Standard)
    # Converts model predictions to relative ranks (0-1 percentile) first, then averages them.
    # Prevents calibration distortion and directly optimizes the ROC-AUC rank-based metric.
    print("\nComputing Rank-Average Blend...")
    rank_oof = np.zeros(len(X_stack))
    rank_test = np.zeros(len(X_test_stack))
    
    for oof_col, test_col in zip(stack_cols, test_stack_cols):
        # We weight rank predictions by model's AUC squared to favor strong models
        w = (individual_aucs[oof_col] ** 2) / sum(auc ** 2 for auc in individual_aucs.values())
        
        # Rank mapping
        r_oof = rankdata(oof[oof_col].values) / len(oof)
        r_test = rankdata(test[test_col].values) / len(test)
        
        rank_oof += w * r_oof
        rank_test += w * r_test
        
    # Calibrate Rank-Average Blend using Logistic Regression (Platt Scaling) to preserve ranking order
    lr_cal = LogisticRegression(random_state=42)
    lr_cal.fit(rank_oof.reshape(-1, 1), y)
    calibrated_rank_oof = lr_cal.predict_proba(rank_oof.reshape(-1, 1))[:, 1]
    calibrated_rank_test = lr_cal.predict_proba(rank_test.reshape(-1, 1))[:, 1]
    
    results["Rank-Average Blend (Logistic Calibrated)"] = {
        "auc": roc_auc_score(y, calibrated_rank_oof),
        "brier": brier_score_loss(y, calibrated_rank_oof),
        "oof": calibrated_rank_oof,
        "test": calibrated_rank_test
    }
    
    # --- Cost-Sensitive Threshold Optimization on final best OOF predictions ---
    best_method = max(results, key=lambda k: results[k]["auc"])
    best_oof_proba = results[best_method]["oof"]
    
    print(f"\nBest ensemble method: {best_method} (AUC: {results[best_method]['auc']:.5f})")
    
    print("\n--- Cost-Sensitive Threshold Optimization ---")
    best_threshold = 0.5
    best_cost = float("inf")
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (best_oof_proba >= threshold).astype(int)
        fn = ((y == 1) & (y_pred == 0)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        cost = 5 * fn + 1 * fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
            
    print(f"Optimal threshold: {best_threshold:.2f} (Expected Loss: {best_cost})")
    
    # --- Generate Submission ---
    final_test_proba = results[best_method]["test"]
    
    submission = pd.DataFrame({
        "ACCOUNT_ID": test["ACCOUNT_ID"],
        "CHURN_PROB": final_test_proba
    })
    submission.to_csv("./predictions.csv", index=False)
    print(f"\nSubmission saved to predictions.csv with CHURN_PROB probabilities.")
    
    # Also save probabilities for SHAP and segment analysis
    proba_df = pd.DataFrame({
        "ACCOUNT_ID": test["ACCOUNT_ID"],
        "churn_probability": final_test_proba
    })
    os.makedirs("./predictions", exist_ok=True)
    proba_df.to_csv("./predictions/test_probabilities.csv", index=False)
    
    # Print comparison summary
    print("\n=== ENSEMBLE COMPARISON SUMMARY ===")
    print(f"{'Method':<35} {'AUC':>8} {'Brier':>8}")
    print("-" * 55)
    for method, vals in results.items():
        print(f"{method:<35} {vals['auc']:>8.5f} {vals['brier']:>8.5f}")

if __name__ == "__main__":
    build_stacking_ensemble()
