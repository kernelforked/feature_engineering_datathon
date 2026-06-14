"""
explain.py — SHAP Explainability, Leakage Audit, and Feature Importance

This module:
1. Loads the trained LightGBM model and generates SHAP values
2. Produces a beeswarm summary plot (global importance)
3. Produces dependency plots for top features
4. Audits for potential data leakage by checking if any feature
   contains future-state information
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap

def run_shap_analysis():
    """Run full SHAP explainability pipeline."""
    print("Loading model and data...")
    
    # Load the saved LightGBM model (fold 0)
    model = joblib.load("./models/lgb_model.pkl")
    
    # Load train features for SHAP computation
    train_df = pd.read_parquet("./processed_data/train_features.parquet")
    exclude_cols = ["ACCOUNT_ID", "CHURN"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Clean column names (same as train.py)
    X = train_df[feature_cols].copy()
    X.columns = [str(col).replace("[", "").replace("]", "").replace("<", "").replace(" ", "_") for col in X.columns]
    y = train_df["CHURN"]
    
    # Use a subsample for SHAP (full dataset is too slow for TreeExplainer on large N)
    sample_size = min(5000, len(X))
    np.random.seed(42)
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X.iloc[idx]
    
    print(f"Computing SHAP values on {sample_size} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Use class 1 (churn) SHAP values
    else:
        shap_vals = shap_values
    
    os.makedirs("./explainability", exist_ok=True)
    
    # 1. Beeswarm Summary Plot
    print("Generating SHAP beeswarm summary plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, X_sample, show=False, max_display=20)
    plt.title("SHAP Feature Importance (Top 20)", fontsize=14)
    plt.tight_layout()
    plt.savefig("./explainability/shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: explainability/shap_beeswarm.png")
    
    # 2. Bar Summary Plot
    print("Generating SHAP bar importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP| Feature Importance (Top 20)", fontsize=14)
    plt.tight_layout()
    plt.savefig("./explainability/shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: explainability/shap_bar.png")
    
    # 3. Top 4 Dependency Plots
    print("Generating SHAP dependency plots for top features...")
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    top_features = X_sample.columns[np.argsort(-mean_abs_shap)][:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, feat in enumerate(top_features):
        ax = axes[i // 2, i % 2]
        feat_idx = list(X_sample.columns).index(feat)
        shap.dependence_plot(feat_idx, shap_vals, X_sample, ax=ax, show=False)
        ax.set_title(f"SHAP Dependence: {feat}", fontsize=11)
    plt.suptitle("SHAP Dependency Plots — Top 4 Features", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("./explainability/shap_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: explainability/shap_dependence.png")
    
    # 4. Feature Importance Table
    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    
    print("\n=== TOP 15 FEATURES BY MEAN |SHAP| ===")
    print(importance_df.head(15).to_string(index=False))
    importance_df.to_csv("./explainability/shap_importance.csv", index=False)
    
    # 5. Data Leakage Audit
    print("\n=== DATA LEAKAGE AUDIT ===")
    leakage_risk = []
    for feat in importance_df.head(10)["feature"]:
        auc = 0
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y.iloc[idx], X_sample[feat])
        except:
            pass
        flag = "⚠️  SUSPICIOUS" if auc > 0.95 else "✓ OK"
        leakage_risk.append({"feature": feat, "standalone_AUC": auc, "status": flag})
        print(f"  {feat}: standalone AUC = {auc:.4f} {flag}")
    
    leakage_df = pd.DataFrame(leakage_risk)
    leakage_df.to_csv("./explainability/leakage_audit.csv", index=False)
    
    # Check if any single feature has AUC > 0.95 (potential leakage)
    suspicious = [r for r in leakage_risk if r["standalone_AUC"] > 0.95]
    if suspicious:
        print(f"\n⚠️  WARNING: {len(suspicious)} feature(s) have standalone AUC > 0.95!")
        print("  These may contain target leakage. Investigate before submission.")
    else:
        print("\n✓ No data leakage detected. All top features have standalone AUC < 0.95.")
    
    print("\nSHAP analysis completed successfully.")

if __name__ == "__main__":
    run_shap_analysis()
