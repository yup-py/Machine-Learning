import os
import pandas as pd
import numpy as np
from src.lock_manager import LockManager, StepAlreadyDone

# Path where we save the final "AI-ready" data
PROCESSED_PATH = "data/processed/"

def _run_feature_logic(X_train, X_test):
    """Internal logic to create new features."""
    # Create copies to avoid 'SettingWithCopy' warnings
    X_train, X_test = X_train.copy(), X_test.copy()

    # 1. LOG TRANSFORMATION (Handling Outliers)
    # np.log1p handles 0 values safely by doing log(1 + x)
    if "surface_m2" in X_train.columns:
        X_train["log_surface"] = np.log1p(X_train["surface_m2"])
        X_test["log_surface"] = np.log1p(X_test["surface_m2"])

    # 2. RATIO: SURFACE PER ROOM
    # Adding +1 to rooms to avoid DivisionByZero errors
    if "surface_m2" in X_train.columns and "rooms" in X_train.columns:
        X_train["surface_per_room"] = X_train["surface_m2"] / (X_train["rooms"] + 1)
        X_test["surface_per_room"] = X_test["surface_m2"] / (X_test["rooms"] + 1)

    # 3. INTERACTION: LUXURY SCORE (Example)
    # High rooms + High bathrooms usually indicates a higher price bracket
    if "rooms" in X_train.columns and "bathrooms" in X_train.columns:
        X_train["room_bath_ratio"] = X_train["bathrooms"] / (X_train["rooms"] + 1)
        X_test["room_bath_ratio"] = X_test["bathrooms"] / (X_test["rooms"] + 1)

    # Save final versions
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train_engineered.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test_engineered.csv"), index=False)
    
    return X_train, X_test

def engineer_features(X_train, X_test, force=False):
    """Main wrapper with Locking mechanism."""
    try:
        with LockManager("feature_engineering", force=force):
            return _run_feature_logic(X_train, X_test)
    except StepAlreadyDone:
        print("[feature_engineering] Already done today — loading engineered files.")
        X_tr = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_engineered.csv"))
        X_te = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test_engineered.csv"))
        return X_tr, X_te