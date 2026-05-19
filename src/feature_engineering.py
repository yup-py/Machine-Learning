import os
import pandas as pd
import numpy as np
from src.lock_manager import LockManager, StepAlreadyDone
from src.logger import get_logger

logger = get_logger()
PROCESSED_PATH = "data/processed/"

def _run_feature_logic(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    
    # 1. HARD OUTBOUND LEAKAGE REMOVAL
    leakage_cols = ['price_per_m2', 'price_da', 'price_dh', 'price']
    X_train = X_train.drop(columns=[c for c in leakage_cols if c in X_train.columns], errors='ignore')
    X_test = X_test.drop(columns=[c for c in leakage_cols if c in X_test.columns], errors='ignore')

    # 2. LOG SCALE NORMALIZATION
    for df in [X_train, X_test]:
        if "surface_m2" in df.columns:
            df["log_surface"] = np.log1p(df["surface_m2"])
        
    # 3. BUILDING RATIO EXPERIMENTS
    for df in [X_train, X_test]:
        # Using safely imputed rooms from the dataset
        df["surface_per_room"] = df["surface_m2"] / (df["rooms"] + 1)
        df["bath_room_ratio"] = df["bathrooms"] / (df["rooms"] + 1)

    # 4. REGIONAL COHORT CHARACTERISTICS (Grouping by City and Category)
    group_cols = ['city', 'category']
    group_medians = X_train.groupby(group_cols)['surface_m2'].median().reset_index()
    group_medians.rename(columns={'surface_m2': 'group_median_surface'}, inplace=True)

    X_train = X_train.merge(group_medians, on=group_cols, how='left')
    X_test = X_test.merge(group_medians, on=group_cols, how='left')

    for df in [X_train, X_test]:
        df['group_median_surface'] = df['group_median_surface'].fillna(X_train['surface_m2'].median())
        df["relative_size_index"] = df["surface_m2"] / (df["group_median_surface"] + 1)
        df.drop(columns=['group_median_surface'], inplace=True)

    # 5. THE BULLETPROOF SANITY MASK (Executed post-encoding)
    # Since LabelEncoder has run, Category 4 is definitely your Terrains/Lands now
    LAND_CAT = 4
    
    for df in [X_train, X_test]:
        if 'category' in df.columns:
            # Locate all rows where category code is 4 (Lands)
            is_land = (df['category'] == LAND_CAT)
            
            # Force structural values to exactly 0 so they are clean
            df.loc[is_land, 'rooms'] = 0.0
            df.loc[is_land, 'bathrooms'] = 0.0
            df.loc[is_land, 'surface_per_room'] = 0.0
            df.loc[is_land, 'bath_room_ratio'] = 0.0

    # 6. EXPORT CLEAN ENGINEERED DATASETS
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    X_train.to_csv(os.path.join(PROCESSED_PATH, "X_train_engineered.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_PATH, "X_test_engineered.csv"), index=False)
    
    logger.info(f"[feature_engineering] Feature step complete. Output shape: {X_train.shape}")
    return X_train, X_test

def engineer_features(X_train, X_test, force=False):
    try:
        with LockManager("feature_engineering", force=force):
            return _run_feature_logic(X_train, X_test)
    except StepAlreadyDone:
        logger.info("[feature_engineering] Pipeline locked — reloading files.")
        X_train = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_engineered.csv"))
        X_test = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test_engineered.csv"))
        return X_train, X_test