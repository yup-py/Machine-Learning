# Daily rotating logger — writes to logs/YYYY-MM-DD.log + console

import os
import sys
import logging
from datetime import date

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config.config import LOGS_PATH


def get_logger(name: str = "ml_pipeline") -> logging.Logger:
    
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # File handler  — one log file per day
    os.makedirs(LOGS_PATH, exist_ok=True)
    log_file = os.path.join(LOGS_PATH, f"{date.today().isoformat()}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialised — file: {log_file}")
    return logger
