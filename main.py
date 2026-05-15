import sys
from src.extraction import extract_obt
from src.preprocessing import preprocess_for_regression # Your Step 3 file
from src.feature_engineering import engineer_features    # Your Step 4 file

def main():
    FORCE = "--force" in sys.argv
    
    print("="*60)
    print("  MOROCCO REAL ESTATE PIPELINE — BRIEF VERSION")
    print(f"  force={FORCE}")
    print("="*60)

    # STEP 1: Extraction
    df_raw = extract_obt(force=FORCE)

    if df_raw is not None:
        # This handles Split, Missing Values, and Encoding
        X_train, X_test, y_train, y_test, encoders, scaler = preprocess_for_regression(df_raw, force=FORCE)

        # we add ratios and log transforms to the split data
        X_train_final, X_test_final = engineer_features(X_train, X_test, force=FORCE)

        print("\n" + "-"*30)
        print(f"Final Train Shape: {X_train_final.shape}")
        print(f"Final Test Shape:  {X_test_final.shape}")
        print("-"*30)

    print("Pipeline execution finished.")

if __name__ == "__main__":
    main()