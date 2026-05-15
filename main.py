import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.logger import get_logger
from src.lock_manager import print_status_report
from src.extraction import extract_obt
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess_for_regression, preprocess_for_classification

logger = get_logger()

# Pass --force as CLI arg to override today's locks and re-run everything
FORCE = "--force" in sys.argv

def main():
    logger.info("=" * 60)
    logger.info("  ML PIPELINE  —  Version 1  (Steps 1-3)")
    logger.info(f"  force={FORCE}")
    logger.info("=" * 60)

    # Today's lock status
    print_status_report()

    # Step 1: Extraction
    logger.info("\n[STEP 1] Extraction")
    df_raw = extract_obt(save_csv=True, force=FORCE)

    # Step 2: Feature Engineering
    logger.info("\n[STEP 2] Feature Engineering")
    df_eng = engineer_features(df_raw, force=FORCE)

    # Step 3: Preprocessing
    logger.info("\n[STEP 3] Preprocessing")

    X_train_r, X_test_r, y_train_r, y_test_r, enc_r, scaler_r = \
        preprocess_for_regression(df_eng, force=FORCE)

    X_train_c, X_test_c, y_train_c, y_test_c, enc_c, scaler_c = \
        preprocess_for_classification(df_eng, force=FORCE)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  DONE — Steps 1-3 completed")
    logger.info("=" * 60)
    logger.info(f"  Raw data : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
    logger.info(f"  Engineered : {df_eng.shape[0]:,} rows × {df_eng.shape[1]} cols")
    logger.info(f"  Regression train / test : {len(X_train_r):,} / {len(X_test_r):,}")
    logger.info(f"  Classification train / test : {len(X_train_c):,} / {len(X_test_c):,}")
    logger.info(f"  Processed data → data/processed/")
    logger.info(f"  Logs → logs/")
    logger.info(f"  Locks → locks/")

    # Final lock report
    print_status_report()

if __name__ == "__main__":
    main()
