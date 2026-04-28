# logger.py

import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Call this once at app startup.
    Returns a configured logger for the application.
    """
    
    Path("logs").mkdir(exist_ok=True)
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger = logging.getLogger("rag_app")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler("logs/app.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    error_handler = logging.FileHandler("logs/errors.log", mode="a", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logging()