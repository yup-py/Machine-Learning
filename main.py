# main.py
import sys
import os
from src.logger import get_logger
from src.extraction import extract_obt
from src.preprocessing import preprocess_for_regression
from src.feature_engineering import engineer_features
from src.train import train_regression_model
from src.train_classification import train_classification_model

logger = get_logger("main_orchestrator")

def run_pipeline(force_all=False):
    logger.info("========== STARTING REAL ESTATE ML PIPELINE ==========")
    
    try:
        # Step 1: Extract Data
        logger.info("--> STEP 1: Extraction")
        df_raw = extract_obt(save_csv=True, force=force_all)
        
        # Step 2: Preprocessing & Split
        logger.info("--> STEP 2: Preprocessing")
        X_train, X_test, y_train, y_test, _, _ = preprocess_for_regression(df_raw, force=force_all)
        
        # Step 3: Feature Engineering
        logger.info("--> STEP 3: Feature Engineering")
        X_train_eng, X_test_eng = engineer_features(X_train, X_test, force=force_all)
        
        # Step 4: Model Training (Regression)
        logger.info("--> STEP 4: Regression Training (Price)")
        reg_model, reg_metrics = train_regression_model(force=force_all)
        
        # Step 5: Model Training (Classification)
        logger.info("--> STEP 5: Classification Training (Category)")
        clf_model, clf_metrics = train_classification_model(force=force_all)
        
        logger.info("========== PIPELINE EXECUTED SUCCESSFULLY ==========")
        
    except Exception as e:
        logger.error(f"Pipeline failed at a critical step: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Set force_all=True to ignore locks and force a full re-run today
    run_pipeline(force_all=False)