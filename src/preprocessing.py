# Step 3 — Post-DB data preparation

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import (
    TARGET_REGRESSION, TARGET_CLASSIFICATION,
    TEST_SIZE, RANDOM_STATE,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    PROCESSED_DATA_PATH,
)
from src.logger import get_logger
from src.lock_manager import LockManager, StepAlreadyDone

logger = get_logger()


# 1. Train / Test split

def split_data(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info(f"[preprocessing] Split → train: {len(X_train):,}  test: {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# 2. Missing values

def handle_missing(X_train, X_test):
    num_cols = [c for c in NUMERICAL_COLS  if c in X_train.columns]
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        X_train[num_cols] = num_imp.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imp.transform(X_test[num_cols])

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imp.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imp.transform(X_test[cat_cols])

    missing = X_train.isnull().sum().sum()
    logger.info(f"[preprocessing] Missing values after imputation: {missing}")
    return X_train, X_test


# 3. Encoding

def encode_categoricals(X_train, X_test):
    encoders = {}
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train.columns]

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    logger.info(f"[preprocessing] Encoded: {cat_cols}")
    return X_train, X_test, encoders


# 4. Scaling

def scale_features(X_train, X_test):
    num_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns]
    scaler   = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols]  = scaler.transform(X_test[num_cols])
    logger.info(f"[preprocessing] Scaled {len(num_cols)} numerical columns.")
    return X_train, X_test, scaler


# 5. SMOTE

def apply_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("[preprocessing] imbalanced-learn not installed. Skipping SMOTE.")
        return X_train, y_train

    counts = y_train.value_counts()
    ratio  = counts.iloc[0] / counts.iloc[-1]
    logger.info(f"[preprocessing] Class distribution:\n{counts.to_string()}")

    if ratio > 3:
        logger.info(f"[preprocessing] Imbalance ratio={ratio:.1f} — applying SMOTE.")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        logger.info(f"[preprocessing] After SMOTE: {len(X_train):,} samples.")
    else:
        logger.info("[preprocessing] Classes balanced — SMOTE not applied.")
    return X_train, y_train


# Save helpers

def _save(X_train, X_test, y_train, y_test, tag):
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    X_train.to_csv(f"{PROCESSED_DATA_PATH}X_train_{tag}.csv", index=False)
    X_test.to_csv( f"{PROCESSED_DATA_PATH}X_test_{tag}.csv",  index=False)
    y_train.to_csv(f"{PROCESSED_DATA_PATH}y_train_{tag}.csv", index=False)
    y_test.to_csv( f"{PROCESSED_DATA_PATH}y_test_{tag}.csv",  index=False)
    logger.info(f"[preprocessing] Saved processed splits → {PROCESSED_DATA_PATH} (tag={tag})")


def _load(tag):
    p = PROCESSED_DATA_PATH
    X_train = pd.read_csv(f"{p}X_train_{tag}.csv")
    X_test  = pd.read_csv(f"{p}X_test_{tag}.csv")
    y_train = pd.read_csv(f"{p}y_train_{tag}.csv").squeeze()
    y_test  = pd.read_csv(f"{p}y_test_{tag}.csv").squeeze()
    logger.info(f"[preprocessing] Reloaded splits from disk (tag={tag})")
    return X_train, X_test, y_train, y_test


# Locked pipelines

def preprocess_for_regression(df: pd.DataFrame, force: bool = False):
    """Full preprocessing pipeline for regression — locked per day."""
    tag = "regression"
    lock_name = f"preprocessing_{tag}"

    try:
        with LockManager(lock_name, force=force):
            logger.info("[preprocessing] === Regression pipeline ===")
            X_train, X_test, y_train, y_test = split_data(df, TARGET_REGRESSION)
            X_train, X_test = handle_missing(X_train, X_test)
            X_train, X_test, encoders  = encode_categoricals(X_train, X_test)
            X_train, X_test, scaler  = scale_features(X_train, X_test)
            _save(X_train, X_test, y_train, y_test, tag)
            return X_train, X_test, y_train, y_test, encoders, scaler

    except StepAlreadyDone:
        logger.info("[preprocessing] Regression splits already done today — reloading.")
        X_train, X_test, y_train, y_test = _load(tag)
        return X_train, X_test, y_train, y_test, {}, None


def preprocess_for_classification(df: pd.DataFrame, force: bool = False):
    # Full preprocessing pipeline for classification — locked per day.
    tag = "classification"
    lock_name = f"preprocessing_{tag}"

    try:
        with LockManager(lock_name, force=force):
            logger.info("[preprocessing] === Classification pipeline ===")
            df_cls = df.drop(columns=[TARGET_REGRESSION], errors="ignore")
            X_train, X_test, y_train, y_test = split_data(df_cls, TARGET_CLASSIFICATION)
            X_train, X_test = handle_missing(X_train, X_test)
            X_train, X_test, encoders = encode_categoricals(X_train, X_test)
            X_train, X_test, scaler  = scale_features(X_train, X_test)
            X_train, y_train = apply_smote(X_train, y_train)
            _save(X_train, X_test, y_train, y_test, tag)
            return X_train, X_test, y_train, y_test, encoders, scaler

    except StepAlreadyDone:
        logger.info("[preprocessing] Classification splits already done today — reloading.")
        X_train, X_test, y_train, y_test = _load(tag)
        return X_train, X_test, y_train, y_test, {}, None


if __name__ == "__main__":
    from src.extraction import load_from_csv
    df = load_from_csv()
    preprocess_for_regression(df)
    preprocess_for_classification(df)
