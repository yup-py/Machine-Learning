# src/optimize.py
import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import MODELS_PATH, PROCESSED_DATA_PATH
from src.logger import get_logger

logger = get_logger()

def optimize_regression_model():
    logger.info("[optimization] === Starting Hyperparameter Tuning ===")
    
    # Load engineered features and scaler
    X_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_train_engineered.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "y_train.csv")).values.ravel()
    scaler = joblib.load(os.path.join(MODELS_PATH, "feature_scaler.pkl"))
    
    X_train_scaled = scaler.transform(X_train)

    # Define the parameter grid to search through
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    base_model = RandomForestRegressor(random_state=42)
    
    # RandomizedSearchCV is faster than GridSearchCV for heavy datasets
    search = RandomizedSearchCV(
        estimator=base_model, 
        param_distributions=param_grid, 
        n_iter=10, # Number of parameter settings sampled
        cv=3,      # 3-fold cross validation
        verbose=2, 
        random_state=42, 
        n_jobs=-1
    )

    logger.info("[optimization] Running RandomizedSearchCV...")
    search.fit(X_train_scaled, y_train)

    best_model = search.best_estimator_
    logger.info(f"[optimization] Best parameters found: {search.best_params_}")

    # Extract Feature Importance
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    print("\n" + "="*40)
    print("   TOP 5 MOST IMPORTANT FEATURES")
    print("="*40)
    print(importance_df.head(5).to_string(index=False))
    print("="*40 + "\n")

    # Save the optimized model
    joblib.dump(best_model, os.path.join(MODELS_PATH, "optimized_regression_model.pkl"))
    logger.info("[optimization] Optimized model saved successfully.")

if __name__ == "__main__":
    optimize_regression_model()