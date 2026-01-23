import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(name: str, level='INFO', log_file=None, format_string=None):
    """Setup a logger with console and optional file output"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str):
    """Get or create a logger with default config"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        from src.config import config
        logger = setup_logger(name, level=config.LOG_LEVEL, log_file=config.LOG_FILE)
    
    return logger