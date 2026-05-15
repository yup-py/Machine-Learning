# Step 2 — Advanced feature engineering

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import PROCESSED_DATA_PATH
from src.logger import get_logger
from src.lock_manager import LockManager, StepAlreadyDone

logger    = get_logger()
LOCK_NAME = "feature_engineering"


# Individual transformations

def log_transform_price(df: pd.DataFrame, col: str = "price_da") -> pd.DataFrame:
    if col in df.columns:
        df[f"log_{col}"] = np.log1p(df[col])
        logger.debug(f"[feature_engineering] Added log_{col}")
    return df


def add_price_per_surface(df: pd.DataFrame) -> pd.DataFrame:
    if "price_da" in df.columns and "surface_m2" in df.columns:
        df["price_per_m2"] = df.apply(
            lambda r: r["price_da"] / r["surface_m2"] if r["surface_m2"] > 0 else np.nan,
            axis=1,
        )
        df["price_per_m2"] = df["price_per_m2"].fillna(df["price_per_m2"].median())
        logger.debug("[feature_engineering] Added / refreshed price_per_m2")
    return df


def add_room_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if "surface_m2" in df.columns and "rooms" in df.columns:
        df["surface_x_rooms"] = df["surface_m2"] * df["rooms"]
        logger.debug("[feature_engineering] Added surface_x_rooms")
    return df


def add_surface_per_room(df: pd.DataFrame) -> pd.DataFrame:
    if "surface_m2" in df.columns and "rooms" in df.columns:
        df["surface_per_room"] = df.apply(
            lambda r: r["surface_m2"] / r["rooms"] if r["rooms"] > 0 else np.nan,
            axis=1,
        )
        df["surface_per_room"] = df["surface_per_room"].fillna(df["surface_per_room"].median())
        logger.debug("[feature_engineering] Added surface_per_room")
    return df


def add_geographic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if "city" in df.columns and "price_da" in df.columns:
        df["city_avg_price"] = df.groupby("city")["price_da"].transform("mean")
        df["city_freq"]      = df.groupby("city")["price_da"].transform("count")
        logger.debug("[feature_engineering] Added city_avg_price, city_freq")
    if "district" in df.columns and "price_da" in df.columns:
        df["district_avg_price"] = df.groupby("district")["price_da"].transform("mean")
        logger.debug("[feature_engineering] Added district_avg_price")
    return df


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    if "property_age_years" in df.columns:
        bins   = [-1, 5, 15, 30, np.inf]
        labels = [0, 1, 2, 3]
        df["age_group"] = pd.cut(df["property_age_years"], bins=bins, labels=labels).astype(int)
        logger.debug("[feature_engineering] Added age_group")
    return df


def add_bathroom_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if "bathrooms" in df.columns and "rooms" in df.columns:
        df["bathroom_ratio"] = df.apply(
            lambda r: r["bathrooms"] / r["rooms"] if r["rooms"] > 0 else 0, axis=1
        )
        logger.debug("[feature_engineering] Added bathroom_ratio")
    return df


def add_floor_indicator(df: pd.DataFrame) -> pd.DataFrame:
    if "floor" in df.columns:
        df["is_ground_floor"] = (df["floor"] == 0).astype(int)
        logger.debug("[feature_engineering] Added is_ground_floor")
    return df


# Full pipeline

def _run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Internal all transformations in order.
    df = df.copy()
    df = log_transform_price(df)
    df = add_price_per_surface(df)
    df = add_room_interactions(df)
    df = add_surface_per_room(df)
    df = add_geographic_indicators(df)
    df = add_age_groups(df)
    df = add_bathroom_ratio(df)
    df = add_floor_indicator(df)

    new_cols = [
        "log_price_da", "price_per_m2", "surface_x_rooms", "surface_per_room",
        "city_avg_price", "city_freq", "district_avg_price",
        "age_group", "bathroom_ratio", "is_ground_floor",
    ]
    added = [c for c in new_cols if c in df.columns]
    logger.info(f"[feature_engineering] {len(added)} new features: {added}")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    out_path = f"{PROCESSED_DATA_PATH}obt_engineered.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"[feature_engineering] Saved → {out_path}")
    return df


_ENGINEERED_PATH = f"{PROCESSED_DATA_PATH}obt_engineered.csv"


def engineer_features(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    Locked per day — reloads the saved CSV if already done today.
    """
    try:
        with LockManager(LOCK_NAME, force=force):
            return _run_feature_engineering(df)

    except StepAlreadyDone:
        logger.info("[feature_engineering] Already ran today — loading saved file.")
        if not os.path.exists(_ENGINEERED_PATH):
            logger.warning("[feature_engineering] Saved file missing — re-running.")
            return _run_feature_engineering(df)
        df_eng = pd.read_csv(_ENGINEERED_PATH)
        logger.info(f"[feature_engineering] Loaded from CSV: {df_eng.shape}")
        return df_eng


if __name__ == "__main__":
    from src.extraction import load_from_csv
    df = load_from_csv()
    df_eng = engineer_features(df)
    print(df_eng.head())
