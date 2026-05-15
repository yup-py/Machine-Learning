# Database connection
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "your_database",   # <-- change this
    "user": "admin",               # <-- change this
    "password": "your_password",   # <-- change this
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
TARGET_REGRESSION      = "price_da"
TARGET_CLASSIFICATION  = "category"
TEST_SIZE              = 0.2
RANDOM_STATE           = 42

# Columns as seen in the OBT table
CATEGORICAL_COLS = ["category", "city", "district"]
NUMERICAL_COLS   = ["surface_m2", "rooms", "bathrooms", "floor",
                    "property_age_years", "price_per_m2"]
