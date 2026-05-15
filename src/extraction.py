# Step 1 — Extract OBT from the Data Warehouse

import os
import sys
import pandas as pd
import psycopg2

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import DB_CONFIG, OBT_TABLE, RAW_DATA_PATH
from src.logger import get_logger
from src.lock_manager import LockManager, StepAlreadyDone

logger = get_logger()

LOCK_NAME = "extraction"


def get_connection():
    # Open a connection to the PostgreSQL database.
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        logger.info("[extraction] Connected to the database.")
        return conn
    except Exception as e:
        logger.error(f"[extraction] Connection failed: {e}")
        raise


def _run_extraction(save_csv: bool) -> pd.DataFrame:
    # Internal — actual extraction logic (called inside the lock)
    conn = get_connection()
    query = f"SELECT * FROM {OBT_TABLE};"
    logger.info(f"[extraction] Query: {query}")

    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    logger.info(f"[extraction] Extracted {len(df):,} rows × {df.shape[1]} columns.")
    logger.debug(f"[extraction] Columns: {list(df.columns)}")

    if save_csv:
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"[extraction] Raw snapshot saved → {RAW_DATA_PATH}")

    return df


def extract_obt(save_csv: bool = True, force: bool = False) -> pd.DataFrame:
    """
    Pull the full OBT table from ml_schema.
    Locked per day — skips if already ran successfully today (unless force=True)

    save_csv : save a local CSV snapshot in data/raw/
    force    : re-run even if today's lock says done
    """
    try:
        with LockManager(LOCK_NAME, force=force):
            df = _run_extraction(save_csv)
        return df

    except StepAlreadyDone:
        logger.info("[extraction] Already ran today — loading local CSV snapshot.")
        return load_from_csv()


def load_from_csv() -> pd.DataFrame:
    # Load the local CSV snapshot without hitting the database.
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"No local snapshot at {RAW_DATA_PATH}. Run extract_obt() first."
        )
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"[extraction] Loaded from CSV: {len(df):,} rows × {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    df = extract_obt(save_csv=True)
    print(df.head())
