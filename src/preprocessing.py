import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import (
    TARGET_REGRESSION, TEST_SIZE, RANDOM_STATE,
    CATEGORICAL_COLS, NUMERICAL_COLS, PROCESSED_DATA_PATH
)
from src.lock_manager import LockManager, StepAlreadyDone
from src.logger import get_logger

logger = get_logger()

def preprocess_for_regression(df: pd.DataFrame, force: bool = False):
    tag = "regression"
    lock_name = f"preprocessing_{tag}"

    try:
        with LockManager(lock_name, force=force):
            logger.info("[preprocessing] === Starting Regression Preprocessing ===")
            
            # 1. DROP STRUCTURAL COLUMNS IRRELEVANT FOR REGRESSION RUN
            cols_to_drop = ['floor', 'property_age_years']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
            logger.info(f"[preprocessing] Dropped optional columns: {cols_to_drop}")

            # 2. CLEANLY DIVIDE TARGET AND FEATURES
            X = df.drop(columns=[TARGET_REGRESSION])
            y = df[TARGET_REGRESSION]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            X_train, X_test = X_train.copy(), X_test.copy()
            
            # Identify Land/Terrain category dynamically. 
            # If your label encoding assigns a different string value later, we handle it pre-encoding here
            # Assuming code 4 or string description matches terrains
            LAND_IDENTIFIERS = [4, "4", "terrain", "terrains", "Terrain"]

            # 3. CONTEXT-AWARE BLANK VALUES HANDLING
            # Explicitly force rooms and bathrooms to 0 for lands before running grouping averages
            for dataset in [X_train, X_test]:
                if 'category' in dataset.columns:
                    is_land = dataset['category'].isin(LAND_IDENTIFIERS)
                    dataset.loc[is_land, ['rooms', 'bathrooms']] = 0

            # Grouped cohort imputation for actual buildings (Apartments, Villas)
            group_cols = ['city', 'category']
            current_num_cols = [c for c in NUMERICAL_COLS if c in X_train.columns]

            for col in current_num_cols:
                # Group medians calculated STRICTLY from training data to stop target leak
                medians = X_train.groupby(group_cols)[col].median().reset_index()
                medians.rename(columns={col: f'median_{col}'}, inplace=True)

                # Merge and impute Train
                X_train = X_train.merge(medians, on=group_cols, how='left')
                X_train[col] = X_train[col].fillna(X_train[f'median_{col}'])
                X_train.drop(columns=[f'median_{col}'], inplace=True)

                # Merge and impute Test using Train baseline
                X_test = X_test.merge(medians, on=group_cols, how='left')
                X_test[col] = X_test[col].fillna(X_test[f'median_{col}'])
                X_test.drop(columns=[f'median_{col}'], inplace=True)

            # Global backup fallback if entire city/category group matches are empty
            X_train[current_num_cols] = X_train[current_num_cols].fillna(X_train[current_num_cols].median())
            X_test[current_num_cols] = X_test[current_num_cols].fillna(X_train[current_num_cols].median())

            # 4. CATEGORICAL TRANSFORMATION (Label Encoding)
            current_cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
            for col in current_cat_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                
                # Safe testing map: assigns -1 if unexpected text label appears in future testing snapshots
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            
            logger.info("[preprocessing] Completed contextual clean and categorical splits successfully.")

            # 5. EXPORT PREPROCESSED RAW FILES
            os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
            X_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "X_train_raw.csv"), index=False)
            X_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "X_test_raw.csv"), index=False)
            y_train.to_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv"), index=False)
            y_test.to_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv"), index=False)

            return X_train, X_test, y_train, y_test, None, None

    except StepAlreadyDone:
        logger.info("[preprocessing] Step locked — loading saved splits.")
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_train_raw.csv"))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_test_raw.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv"))
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv"))
        return X_train, X_test, y_train, y_test, None, None