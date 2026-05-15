import os
import pandas as pd
import numpy as np
from src.lock_manager import LockManager, StepAlreadyDone
from src.logger import get_logger

logger = get_logger()
PROCESSED_PATH = "data/processed/"

def _run_feature_logic(X_train, X_test):
    """
    Final feature engineering logic:
    - Keeps terrains/farms safely via Log Transforms.
    - Creates group-based relative metrics (City/Category).
    - Removes all potential Target Leakage.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    
    # 1. REMOVE LEAKAGE
    # We must ensure the price is never used to create features.
    leakage_cols = ['price_per_m2', 'price_da', 'price_dh', 'price']
    X_train = X_train.drop(columns=[c for c in leakage_cols if c in X_train.columns], errors='ignore')
    X_test = X_test.drop(columns=[c for c in leakage_cols if c in X_test.columns], errors='ignore')

    # 2. LOG TRANSFORMATION (Handling Terrains & Large Estates)
    # This turns your 73,000 m2 into ~11.1, making it mathematically stable.
    for df in [X_train, X_test]:
        if "surface_m2" in df.columns:
            df["log_surface"] = np.log1p(df["surface_m2"])
        
    # 3. DOMAIN-SPECIFIC FEATURES (Moroccan Market Logic)
    for df in [X_train, X_test]:
        # Boolean flag: Is it a massive property (Terrain/Farm)?
        df['is_massive'] = (df['surface_m2'] > 1500).astype(int)
        
        # Interaction: How 'dense' is the property?
        df["surface_per_room"] = df["surface_m2"] / (df["rooms"] + 1)
        
        # Luxury Indicator: High bathroom-to-room ratio
        df["bath_room_ratio"] = df["bathrooms"] / (df["rooms"] + 1)

    # 4. GROUPED STATISTICAL FEATURES (City & Category)
    # We want the model to know if a house is 'big' compared to its peers.
    group_cols = ['city', 'category']
    
    # Calculate medians from Train set only to avoid leakage
    group_medians = X_train.groupby(group_cols)['surface_m2'].median().reset_index()
    group_medians.columns = group_cols + ['group_median_surface']

    # Merge statistics back to both sets
    X_train = X_train.merge(group_medians, on=group_cols, how='left')
    X_test = X_test.merge(group_medians, on=group_cols, how='left')

    # Calculate Relative Size
    for df in [X_train, X_test]:
        # Fallback if a category/city combo is missing
        df['group_median_surface'] = df['group_median_surface'].fillna(X_train['surface_m2'].median())
        
        # Ratio: 1.0 means average for that city/category, 2.0 means double the average size
        df["relative_size_index"] = df["surface_m2"] / (df["group_median_surface"] + 1)
        
        # Cleanup helper column
        df.drop(columns=['group_median_surface'], inplace=True)

    # 5. SAVE FINAL DATASETS
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train_engineered.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test_engineered.csv"), index=False)
    
    logger.info(f"[feature_engineering] Completed. Train shape: {X_train.shape}")
    return X_train, X_test

def engineer_features(X_train, X_test, force=False):
    """Wrapper with LockManager for daily pipeline execution."""
    try:
        with LockManager("feature_engineering", force=force):
            return _run_feature_logic(X_train, X_test)
    except StepAlreadyDone:
        logger.info("[feature_engineering] Loading existing engineered data.")
        X_tr = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_engineered.csv"))
        X_te = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test_engineered.csv"))
        return X_tr, X_te