"""Logging utilities for the active loop system"""

import logging
import os
from typing import Optional


def setup_logging(
    logger_name: str,
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    propagate: bool = False
) -> logging.Logger:
    """Set up logging configuration
    
    Args:
        logger_name: Name of the logger
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file, if None, only logs to console
        propagate: Whether to propagate to the root logger
        
    Returns:
        Configured logger instance
    """
    # Convert log level string to numeric value
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    logger.handlers = []  # Clear existing handlers
    
    # Prevent propagation to the root logger
    logger.propagate = propagate
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Ensure the log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger 