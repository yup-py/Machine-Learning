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
            logger.info("[preprocessing] === Regression pipeline ===")
            
            # 1. DROP UNNECESSARY COLUMNS
            cols_to_drop = ['floor', 'property_age_years']
            df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
            logger.info(f"[preprocessing] Dropped columns: {cols_to_drop}")

            # 2. SPLIT
            X = df.drop(columns=[TARGET_REGRESSION])
            y = df[TARGET_REGRESSION]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            # 3. GROUPED IMPUTATION (Based on City and Category)
            group_cols = ['city', 'category']
            current_num_cols = [c for c in NUMERICAL_COLS if c in X_train.columns]

            for col in current_num_cols:
                # A. Calculate medians on TRAIN
                # This creates a small table of medians for every City/Category combo
                medians = X_train.groupby(group_cols)[col].median().reset_index()
                medians.columns = group_cols + [f'median_{col}']

                # B. Fill Train
                X_train[col] = X_train[col].fillna(
                    X_train.groupby(group_cols)[col].transform('median')
                )
                
                # C. Fill Test using the 'medians' table we built from Train
                # We merge the medians into Test, fill the NaNs, then drop the extra column
                X_test = X_test.merge(medians, on=group_cols, how='left')
                X_test[col] = X_test[col].fillna(X_test[f'median_{col}'])
                X_test = X_test.drop(columns=[f'median_{col}'])

            # Global fallback for any remaining NaNs
            X_train[current_num_cols] = X_train[current_num_cols].fillna(X_train[current_num_cols].median())
            X_test[current_num_cols] = X_test[current_num_cols].fillna(X_train[current_num_cols].median())
            
            logger.info(f"[preprocessing] Grouped imputation (Merge method) done for: {current_num_cols}")
            
            # 4. CATEGORICAL ENCODING
            current_cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]
            for col in current_cat_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = X_test[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            logger.info(f"[preprocessing] Encoded categorical columns: {current_cat_cols}")

            # 5. SAVE RAW PROCESSED SPLITS
            os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
            X_train.to_csv(f"{PROCESSED_DATA_PATH}X_train_raw.csv", index=False)
            X_test.to_csv(f"{PROCESSED_DATA_PATH}X_test_raw.csv", index=False)
            y_train.to_csv(f"{PROCESSED_DATA_PATH}y_train.csv", index=False)
            y_test.to_csv(f"{PROCESSED_DATA_PATH}y_test.csv", index=False)

            return X_train, X_test, y_train, y_test, None, None

    except StepAlreadyDone:
        logger.info("[preprocessing] Step already completed — reloading splits.")
        X_train = pd.read_csv(f"{PROCESSED_DATA_PATH}X_train_raw.csv")
        X_test = pd.read_csv(f"{PROCESSED_DATA_PATH}X_test_raw.csv")
        y_train = pd.read_csv(f"{PROCESSED_DATA_PATH}y_train.csv").squeeze()
        y_test = pd.read_csv(f"{PROCESSED_DATA_PATH}y_test.csv").squeeze()
        return X_train, X_test, y_train, y_test, None, None