import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Establish paths to config variables
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import MODELS_PATH, PROCESSED_DATA_PATH
from src.lock_manager import LockManager, StepAlreadyDone
from src.logger import get_logger

logger = get_logger()

def train_regression_model(force: bool = False):
    lock_name = "model_training_regression"
    
    try:
        with LockManager(lock_name, force=force):
            logger.info("[training] === Starting Regression Model Training ===")
            
            # 1. LOAD CLEAN ENGINEERED FEATURES AND TARGETS
            X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_train_engineered.csv"))
            X_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_test_engineered.csv"))
            y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv")).values.ravel()
            y_test = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_test.csv")).values.ravel()
            
            logger.info(f"[training] Training features shape: {X_train.shape}")
            
            # 2. FEATURE STANDARDISATION
            # Scaling ensures numerical stability for machine learning algorithms
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 3. TRAIN REGRESSION MODEL
            logger.info("[training] Fitting Random Forest Regressor...")
            model = RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # 4. EVALUATE PERFORMANCES
            predictions = model.predict(X_test_scaled)
            
            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            metrics = {
                "MAE": round(mae, 2),
                "MSE": round(mse, 2),
                "RMSE": round(rmse, 2),
                "R2_Score": round(r2, 4)
            }
            
            print("\n" + "="*40)
            print("   REGRESSION MODEL METRICS RESULTS")
            print("="*40)
            print(f"MAE      (Mean Absolute Error):   {metrics['MAE']} DH")
            print(f"MSE      (Mean Squared Error) :   {metrics['MSE']}")
            print(f"RMSE     (Root Mean Sq. Error):   {metrics['RMSE']} DH")
            print(f"R² Score (Coefficient of Det.):   {metrics['R2_Score']}")
            print("="*40 + "\n")
            
            # 5. AUTOMATION: EXPORT TRAINED MODELS & METRICS ARTIFACTS
            os.makedirs(MODELS_PATH, exist_ok=True)
            
            # Save artifacts to drive
            joblib.dump(model, os.path.join(MODELS_PATH, "regression_model.pkl"))
            joblib.dump(scaler, os.path.join(MODELS_PATH, "feature_scaler.pkl"))
            
            with open(os.path.join(MODELS_PATH, "regression_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
            logger.info("[training] Model, Scaler, and Metrics exported successfully.")
            return model, metrics

    except StepAlreadyDone:
        logger.info("[training] Step locked — Model training already evaluated.")
        return None, None

if __name__ == "__main__":
    train_regression_model(force=True)