# config/config.py
import os
from dotenv import load_dotenv

# Load variables from .env file (if you have one)
load_dotenv()

# --- DATABASE CONFIGURATION ---
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "real_estate_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "admin"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432")
}
OBT_TABLE = "ml_schema.obt_annonces"
# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "obt_snapshot.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "models")
LOGS_PATH = os.path.join(BASE_DIR, "logs")

# --- ML PARAMETERS ---
TARGET_REGRESSION = "price_dh"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Define which columns are numerical vs categorical
NUMERICAL_COLS = ["surface_m2", "rooms", "bathrooms"]
CATEGORICAL_COLS = ['city', 'category', 'district']