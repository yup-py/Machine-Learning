# Real Estate ML Pipeline 🏠

This project is an automated Machine Learning (ML) pipeline designed to process real estate data from a database to predictive modeling. It streamlines the workflow from data extraction to price prediction and property classification.

## Project Goals

The project aims to transform raw real estate listings into an intelligent tool capable of:

1. **Price Prediction (Regression):** Estimating the market value of properties.
2. **Property Classification:** Automatically identifying property types (e.g., apartment, villa).
3. **Process Automation:** Handling the full Data Engineering and ML lifecycle (ETL) automatically.

## Pipeline Architecture

The system follows a modular 5-step workflow to ensure reproducibility and prevent data leakage:

1. **Extraction:** Connects to PostgreSQL and pulls raw data, saving it locally.
2. **Preprocessing:** Cleans data, removes irrelevant columns, and performs smart imputation.
3. **Feature Engineering:** Creates derived features (e.g., `log_surface`) to improve model performance.
4. **Regression Training:** Uses `RandomForestRegressor` to predict prices.
5. **Classification Training:** Uses `GradientBoostingClassifier` combined with `SMOTE` to handle class imbalances.

## Key Features

* **Smart Locking System:** Uses a custom `LockManager` to prevent redundant daily executions, saving database resources.
* **Data Leakage Prevention:** All preprocessing steps (like median imputation and encoding) are calculated strictly on training data and applied to test data.
* **Robust Encoding:** Handles "unseen" categorical values in testing sets by assigning them a safe `-1` label instead of crashing.
* **Automated Logging:** Comprehensive logging for every step, stored in `logs/` for easy debugging.

## Requirements

To run this project, ensure you have the following installed:

**Bash**

```
pip install -r requirements.txt
```

*(Tested with Python 3.8+)*

## How to Run

Execute the pipeline by running the main entry point:

**Bash**

```
python main.py
```

*Note: If you need to force a re-extraction (e.g., after database updates), you can delete the `locks/` folder.*

## Tech Stack

* **Database:** PostgreSQL (via `psycopg3`).
* **Data Manipulation:** `pandas`, `numpy`.
* **ML:** `scikit-learn`, `imbalanced-learn`.
* **Workflow:** Custom `LockManager` and logging system.

*Developed as an automated AI-driven solution for real estate market analysis.*
