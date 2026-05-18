import sys
import os
import pandas as pd
from src.extraction import extract_obt
from src.preprocessing import preprocess_for_regression # Your Step 3 file
from src.feature_engineering import engineer_features    # Your Step 4 file
from src.train import train_regression_model            # Your Step 5 file

def main():
    FORCE = "--force" in sys.argv
    
    print("="*60)
    print("  MOROCCO REAL ESTATE PIPELINE — BRIEF VERSION")
    print(f"  force={FORCE}")
    print("="*60)

    # STEP 1: Extraction (Database execution with local file fallback)
    df_raw = None
    try:
        df_raw = extract_obt(force=FORCE)
    except Exception as e:
        print(f"\n[Database Offline] Using local file snapshot instead: {e}")
        if os.path.exists("data/raw/obt_raw.csv"):
            df_raw = pd.read_csv("data/raw/obt_raw.csv")
        else:
            print("[Error] data/raw/obt_raw.csv not found!")
            return
    
    if df_raw is not None:
        # This handles Split, Missing Values, and Encoding
        X_train, X_test, y_train, y_test, encoders, scaler = preprocess_for_regression(df_raw, force=FORCE)

        # We add ratios and log transforms to the split data
        X_train_final, X_test_final = engineer_features(X_train, X_test, force=FORCE)

        print("\n" + "-"*30)
        print(f"Final Train Shape: {X_train_final.shape}")
        print(f"Final Test Shape:  {X_test_final.shape}")
        print("-"*30)

        # STEP 4: Train and Evaluate Regression Models
        train_regression_model(force=FORCE)

    print("Pipeline execution finished.")

if __name__ == "__main__":
    main()