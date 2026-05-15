import os
from dotenv import load_dotenv

# 1. This line loads your .env file into the system
load_dotenv()

# 2. These lines pull the actual values from your .env
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

SCHEMA = "ml_schema"
OBT_TABLE = "ml_schema.obt_annonces"

# Paths
RAW_DATA_PATH       = "data/raw/obt_raw.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODELS_PATH         = "models/"
LOCKS_PATH          = "locks/"
LOGS_PATH           = "logs/"

# ML settings
TARGET_REGRESSION      = "price_dh"
TARGET_CLASSIFICATION  = "category"
TEST_SIZE              = 0.2
RANDOM_STATE           = 42

# Columns
CATEGORICAL_COLS = ["category", "city", "district"]
NUMERICAL_COLS   = [
    "surface_m2", "rooms", "bathrooms", "floor", 
    "property_age_years", "surface_per_room", 
    "city_avg_price", "district_avg_price"
]