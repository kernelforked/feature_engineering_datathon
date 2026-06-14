import os
import gc
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, brier_score_loss, roc_curve
import lightgbm as lgb
import xgboost as xgb

# Try importing CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("\n[Warning] CatBoost not installed. CatBoost will be skipped in the model zoo.")
    print("To install, run: pip install catboost")

# --- Pipeline Constants ---
# Default to 10 folds for maximum validation accuracy on PC
N_SPLITS = 10
RANDOM_STATE = 42

def load_data():
    """Load the engineered features."""
    train_df = pd.read_parquet("./processed_data/train_features.parquet")
    test_df = pd.read_parquet("./processed_data/test_features.parquet")
    return train_df, test_df


def prepare_inputs(train_df, test_df):
    """Separate features and target, and drop unneeded ID columns."""
    exclude_cols = ["ACCOUNT_ID", "CHURN"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X = train_df[feature_cols].copy()
    y = train_df["CHURN"].copy()
    X_test = test_df[feature_cols].copy()
    
    # Ensure all column names are strings and have no special characters
    clean_cols = [str(col).replace("[", "").replace("]", "").replace("<", "").replace(" ", "_") for col in X.columns]
    X.columns = clean_cols
    X_test.columns = clean_cols
    
    print(f"Features count: {len(X.columns)}")
    return X, y, X_test, X.columns.tolist()


def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    """Compute standard classification metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return {
        "AUC": auc,
        "Brier": brier,
        "F1": f1,
        "Recall": recall,
        "Precision": precision
    }


def get_xgb_device_params():
    """Detect if GPU is available and return appropriate XGBoost parameters."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [CPU] PyTorch detected no CUDA GPU. Using CPU for XGBoost.")
            return {"n_jobs": -1, "tree_method": "hist"}
    except ImportError:
        pass
    
    try:
        clf = xgb.XGBClassifier(tree_method="gpu_hist", n_estimators=1)
        clf.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        print("  [GPU] XGBoost gpu_hist test passed. Enabling GPU.")
        return {"tree_method": "gpu_hist"}
    except Exception as e:
        print(f"  [CPU] XGBoost GPU test failed ({e}). Using CPU.")
        return {"n_jobs": -1, "tree_method": "hist"}


def get_lgb_device_params():
    """Detect if GPU is available and return appropriate LightGBM parameters."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [CPU] PyTorch detected no CUDA GPU. Using CPU for LightGBM.")
            return {"n_jobs": -1}
    except ImportError:
        pass

    try:
        model = lgb.LGBMClassifier(device="gpu", n_estimators=1)
        model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        print("  [GPU] LightGBM GPU training test passed. Enabling GPU.")
        return {"device": "gpu"}
    except Exception:
        pass
        
    try:
        model = lgb.LGBMClassifier(device="cuda", n_estimators=1)
        model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10))
        print("  [GPU] LightGBM CUDA training test passed. Enabling GPU.")
        return {"device": "cuda"}
    except Exception:
        pass
        
    print("  [CPU] No GPU detected for LightGBM. Using CPU.")
    return {"n_jobs": -1}


def get_catboost_device_params():
    """Detect if GPU is available and return appropriate CatBoost parameters."""
    if not CATBOOST_AVAILABLE:
        return {}
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [CPU] PyTorch detected no CUDA GPU. Using CPU for CatBoost.")
            return {"thread_count": -1}
    except ImportError:
        pass

    try:
        model = CatBoostClassifier(task_type="GPU", iterations=1)
        model.fit(np.random.rand(10, 2), np.random.randint(0, 2, 10), verbose=False)
        print("  [GPU] CatBoost GPU training test passed. Enabling GPU.")
        return {"task_type": "GPU"}
    except Exception:
        pass
        
    print("  [CPU] No GPU detected for CatBoost. Using CPU.")
    return {"thread_count": -1}


def train_lgb(X, y, X_test, cv, class_weight=None):
    """Train LightGBM with Stratified CV (deeper search space & GPU support)."""
    print("\nTraining LightGBM...")
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    scale_pos_weight = None
    if class_weight == "balanced":
        scale_pos_weight = (len(y) - y.sum()) / y.sum()
        print(f"  Using scale_pos_weight: {scale_pos_weight:.4f}")
        
    gpu_params = get_lgb_device_params()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Deeper tree parameters for high performance
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "max_depth": 8,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_STATE + fold,
            "verbose": -1
        }
        params.update(gpu_params)
        
        if scale_pos_weight:
            params["scale_pos_weight"] = scale_pos_weight
            
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / cv.n_splits
        models.append(model)
        
    metrics = evaluate_predictions(y, oof_preds)
    print(f"LightGBM CV Results: AUC={metrics['AUC']:.5f}, Brier={metrics['Brier']:.5f}")
    return oof_preds, test_preds, models


def train_xgb(X, y, X_test, cv, class_weight=None):
    """Train XGBoost with Stratified CV (deeper search space & GPU support)."""
    print("\nTraining XGBoost...")
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    scale_pos_weight = None
    if class_weight == "balanced":
        scale_pos_weight = (len(y) - y.sum()) / y.sum()
        print(f"  Using scale_pos_weight: {scale_pos_weight:.4f}")
        
    gpu_params = get_xgb_device_params()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 1500,
            "learning_rate": 0.03,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "random_state": RANDOM_STATE + fold,
            "early_stopping_rounds": 100
        }
        params.update(gpu_params)
        
        if scale_pos_weight:
            params["scale_pos_weight"] = scale_pos_weight
            
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / cv.n_splits
        models.append(model)
        
    metrics = evaluate_predictions(y, oof_preds)
    print(f"XGBoost CV Results: AUC={metrics['AUC']:.5f}, Brier={metrics['Brier']:.5f}")
    return oof_preds, test_preds, models


def train_catboost(X, y, X_test, cv, class_weight=None):
    """Train CatBoost with Stratified CV (GPU-enabled, high performance)."""
    if not CATBOOST_AVAILABLE:
        return None, None, None
        
    print("\nTraining CatBoost...")
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    models = []
    
    auto_class_weights = None
    if class_weight == "balanced":
        auto_class_weights = "Balanced"
        print("  Using auto_class_weights: Balanced")
        
    gpu_params = get_catboost_device_params()
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        params = {
            "iterations": 1500,
            "learning_rate": 0.03,
            "depth": 7,
            "eval_metric": "AUC",
            "random_seed": RANDOM_STATE + fold,
            "l2_leaf_reg": 5,
            "early_stopping_rounds": 100
        }
        params.update(gpu_params)
        
        if auto_class_weights:
            params["auto_class_weights"] = auto_class_weights
            
        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            verbose=False
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / cv.n_splits
        models.append(model)
        
    metrics = evaluate_predictions(y, oof_preds)
    print(f"CatBoost CV Results: AUC={metrics['AUC']:.5f}, Brier={metrics['Brier']:.5f}")
    return oof_preds, test_preds, models





def plot_curves(y_true, oof_dict):
    """Plot AUC-ROC curves for all models in the zoo."""
    plt.figure(figsize=(10, 8))
    for model_name, oof_preds in oof_dict.items():
        if oof_preds is not None:
            fpr, tpr, _ = roc_curve(y_true, oof_preds)
            auc = roc_auc_score(y_true, oof_preds)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
        
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model Zoo ROC Curves (Out-of-Fold)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    os.makedirs("./plots", exist_ok=True)
    plt.savefig("./plots/roc_curves.png", dpi=300, bbox_inches="tight")
    print("ROC curve comparison plot saved to plots/roc_curves.png.")
    plt.close()


def main():
    train_df, test_df = load_data()
    X, y, X_test, feature_names = prepare_inputs(train_df, test_df)
    
    # Free the raw DataFrames to preserve RAM
    train_account_ids = train_df["ACCOUNT_ID"].copy()
    test_account_ids = test_df["ACCOUNT_ID"].copy()
    del train_df, test_df
    gc.collect()
    
    # N-Fold Stratified CV
    print(f"Using {N_SPLITS}-Fold Stratified Cross-Validation.")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    # Train Model Zoo — GBDT-only (Iteration 2: pruned via empirical AUC analysis)
    # Removed: Random Forest (CPU-only, AUC=0.98059, degrades ensemble)
    # Removed: MLP Classifier (CPU-only, AUC=0.97912, degrades ensemble)
    # Removed: Logistic Regression (AUC=0.97808, degrades ensemble)
    # Kept: LightGBM, XGBoost, CatBoost (all GPU-capable, AUC>0.9816)
    print("\n=== Training GBDT Model Zoo (LightGBM + XGBoost + CatBoost) ===")
    
    lgb_oof, lgb_test, lgb_models = train_lgb(X, y, X_test, cv, class_weight="balanced")
    gc.collect()
    
    xgb_oof, xgb_test, xgb_models = train_xgb(X, y, X_test, cv, class_weight="balanced")
    gc.collect()
    
    if CATBOOST_AVAILABLE:
        cat_oof, cat_test, cat_models = train_catboost(X, y, X_test, cv, class_weight="balanced")
        gc.collect()
    else:
        cat_oof, cat_test, cat_models = None, None, None
    
    # Plot and save curves
    oof_dict = {
        "LightGBM": lgb_oof,
        "XGBoost": xgb_oof,
    }
    if CATBOOST_AVAILABLE:
        oof_dict["CatBoost"] = cat_oof
        
    plot_curves(y, oof_dict)
    
    # Save Out-of-fold and test predictions for stacking meta-model
    os.makedirs("./predictions", exist_ok=True)
    
    oof_payload = {
        "ACCOUNT_ID": train_account_ids,
        "CHURN": y,
        "lgb_oof": lgb_oof,
        "xgb_oof": xgb_oof,
    }
    if CATBOOST_AVAILABLE:
        oof_payload["cat_oof"] = cat_oof
        
    oof_df = pd.DataFrame(oof_payload)
    oof_df.to_parquet("./predictions/oof_predictions.parquet")
    
    test_payload = {
        "ACCOUNT_ID": test_account_ids,
        "lgb_test": lgb_test,
        "xgb_test": xgb_test,
    }
    if CATBOOST_AVAILABLE:
        test_payload["cat_test"] = cat_test
        
    test_preds_df = pd.DataFrame(test_payload)
    test_preds_df.to_parquet("./predictions/test_predictions.parquet")
    
    os.makedirs("./models", exist_ok=True)
    joblib.dump(lgb_models[0], "./models/lgb_model.pkl")
    joblib.dump(xgb_models[0], "./models/xgb_model.pkl")
    if CATBOOST_AVAILABLE:
        joblib.dump(cat_models[0], "./models/cat_model.pkl")
        
    print("\nModel training phase completed successfully. All predictions saved.")


if __name__ == "__main__":
    main()
