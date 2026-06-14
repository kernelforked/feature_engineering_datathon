#!/usr/bin/env python3
"""
run_pipeline.py — Full end-to-end pipeline runner.

Executes: features.py → train.py → ensemble.py
Generates: predictions.csv (with CHURN_PROB probabilities)

Usage:
    python3 run_pipeline.py
"""
import time
import subprocess
import sys

def run_step(script_name, description):
    """Run a pipeline step and capture output."""
    print(f"\n{'='*60}")
    print(f"  STEP: {description}")
    print(f"  Script: {script_name}")
    print(f"{'='*60}\n")
    
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,  # Stream output live
        text=True
    )
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {script_name} exited with code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n✅ {description} completed in {elapsed:.1f}s")

def main():
    total_start = time.time()
    
    print("=" * 60)
    print("  FictiPay Datathon — Full Pipeline Runner")
    print("  Dataset: Jan + Feb + March (ALL data)")
    print("=" * 60)
    
    # Step 1: Feature Engineering (includes March transactions now)
    run_step("features.py", "Feature Engineering (Jan+Feb+March)")
    
    # Step 2: Model Training (3 GBDT models x 10 folds)
    run_step("train.py", "GBDT Model Zoo Training (LGB/XGB/CatBoost)")
    
    # Step 3: Stacking Ensemble + Calibration + Submission
    run_step("ensemble.py", "Stacking Ensemble + Calibration → predictions.csv")
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {total_elapsed/60:.1f} minutes")
    print(f"  Output: predictions.csv (CHURN_PROB format)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
