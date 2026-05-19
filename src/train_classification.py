import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Path configurations to root parameters
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import MODELS_PATH, PROCESSED_DATA_PATH
from src.lock_manager import LockManager, StepAlreadyDone
from src.logger import get_logger

logger = get_logger()

def train_classification_model(force: bool = False):
    lock_name = "model_training_classification"
    
    try:
        with LockManager(lock_name, force=force):
            logger.info("[classification] === Starting Classification Model Training ===")
            
            # 1. LOAD CLEAN ENGINEERED SNAPSHOTS
            # We use the engineered datasets as our feature foundation
            X_train_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_train_engineered.csv"))
            X_test_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "X_test_engineered.csv"))
            
            # For classification, our target variable is 'category'
            if 'category' not in X_train_df.columns:
                logger.error("[classification] 'category' missing from engineered features!")
                return None, None
                
            y_train = X_train_df['category'].astype(int).values
            y_test = X_test_df['category'].astype(int).values
            
            # Remove target class and pricing elements to prevent leakage
            cols_to_drop = ['category', 'price_dh', 'price']
            X_train = X_train_df.drop(columns=[c for c in cols_to_drop if c in X_train_df.columns], errors='ignore')
            X_test = X_test_df.drop(columns=[c for c in cols_to_drop if c in X_test_df.columns], errors='ignore')
            
            logger.info(f"[classification] Class Distribution before SMOTE: {np.bincount(y_train)}")
            
            # 2. FEATURE STANDARDIZATION
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 3. BALANCING DATASET VIA SMOTE
            # This generates synthetic rows for minority categories (like farms or rare property types)
            logger.info("[classification] Applying SMOTE to balance class variables...")
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
            logger.info(f"[classification] Class Distribution after SMOTE: {np.bincount(y_train_resampled)}")
            
            # 4. TRAINING THE COMPONENT CLASSIFIER
            logger.info("[classification] Training Gradient Boosting Classifier...")
            clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_resampled, y_train_resampled)
            
            # 5. EVALUATE PREDICTION PERFORMANCE INDICES
            predictions = clf.predict(X_test_scaled)
            
            # Calculate metrics using macro averaging to handle multi-class targets safely
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='macro', zero_division=0)
            recall = recall_score(y_test, predictions, average='macro', zero_division=0)
            f1 = f1_score(y_test, predictions, average='macro', zero_division=0)
            
            metrics = {
                "Accuracy": round(accuracy, 4),
                "Precision_Macro": round(precision, 4),
                "Recall_Macro": round(recall, 4),
                "F1_Score_Macro": round(f1, 4)
            }
            
            print("\n" + "="*40)
            print("   CLASSIFICATION MODEL PERFORMANCE")
            print("="*40)
            print(f"Accuracy :   {metrics['Accuracy'] * 100:.2f}%")
            print(f"Precision:   {metrics['Precision_Macro'] * 100:.2f}%")
            print(f"Recall   :   {metrics['Recall_Macro'] * 100:.2f}%")
            print(f"F1-Score :   {metrics['F1_Score_Macro'] * 100:.2f}%")
            print("="*40 + "\n")
            
            # 6. EXPORT CLASSIFIER UTILITIES
            os.makedirs(MODELS_PATH, exist_ok=True)
            joblib.dump(clf, os.path.join(MODELS_PATH, "property_classifier.pkl"))
            joblib.dump(scaler, os.path.join(MODELS_PATH, "classification_scaler.pkl"))
            
            with open(os.path.join(MODELS_PATH, "classification_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
                
            logger.info("[classification] Classifier pipeline artifacts exported cleanly.")
            return clf, metrics

    except StepAlreadyDone:
        logger.info("[classification] Step locked — Classification model evaluation already cached.")
        return None, None