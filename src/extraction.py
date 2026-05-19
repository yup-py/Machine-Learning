import os
import sys
import pandas as pd
import psycopg
from typing import Optional

# Import internal project modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import DB_CONFIG, OBT_TABLE, RAW_DATA_PATH
from src.logger import get_logger
from src.lock_manager import LockManager, StepAlreadyDone

logger = get_logger()
LOCK_NAME = "extraction"

def get_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        # psycopg 3 connection string/params
        return psycopg.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"[extraction] Database connection failed: {e}")
        raise

def _run_extraction(save_csv: bool) -> pd.DataFrame:
    """Performs the database query and saves the results."""
    logger.info(f"[extraction] Querying table: {OBT_TABLE}")
    
    with get_connection() as conn:
        df = pd.read_sql(f"SELECT * FROM {OBT_TABLE};", conn)

    logger.info(f"[extraction] Success: {len(df):,} rows extracted.")

    if save_csv:
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        logger.info(f"[extraction] Data persisted to {RAW_DATA_PATH}")

    return df

def extract_obt(save_csv: bool = True, force: bool = False) -> pd.DataFrame:
    """
    Main entry point to retrieve the OBT data. 
    Implements a lock mechanism to prevent redundant daily re-extractions.
    """
    try:
        with LockManager(LOCK_NAME, force=force):
            return _run_extraction(save_csv)
    except StepAlreadyDone:
        logger.info("[extraction] Lock detected. Loading from local snapshot.")
        return _load_from_csv()

def _load_from_csv() -> pd.DataFrame:
    """Private helper to load the local data snapshot."""
    if not os.path.exists(RAW_DATA_PATH):
        logger.error("[extraction] Snapshot requested but not found.")
        raise FileNotFoundError(f"No snapshot found at {RAW_DATA_PATH}")
    
    return pd.read_csv(RAW_DATA_PATH)

if __name__ == "__main__":
    try:
        data = extract_obt()
        print(data.head())
    except Exception as e:
        logger.critical(f"[extraction] Pipeline crashed: {e}")