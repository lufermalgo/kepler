"""
Logging utilities for Kepler framework

Provides structured logging with file output and console formatting.
"""

import logging
import os
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler


class KeplerLogger:
    """Centralized logging for Kepler framework"""
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def get_logger(cls, name: str = "kepler") -> logging.Logger:
        """Get a configured logger instance"""
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
            if not cls._configured:
                cls._configure_logging()
        return cls._loggers[name]
    
    @classmethod
    def _configure_logging(cls):
        """Configure logging with file and console handlers"""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # File handler with detailed formatting
        file_handler = logging.FileHandler(logs_dir / "kepler.log")
        file_formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler with Rich formatting - notebook-friendly mode
        notebook_mode = os.getenv('KEPLER_NOTEBOOK_MODE', 'true').lower() == 'true'
        
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=not notebook_mode  # Simplified for notebooks
        )
        console_formatter = logging.Formatter(
            fmt="%(message)s",
            datefmt="[%X]"
        )
        console_handler.setFormatter(console_formatter)
        
        # Set console level based on mode
        if notebook_mode:
            console_handler.setLevel(logging.WARNING)  # Only warnings and errors in notebooks
        else:
            console_handler.setLevel(logging.INFO)  # Normal verbosity in CLI
        
        # Configure root kepler logger
        root_logger = logging.getLogger("kepler")
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        root_logger.propagate = False
        
        # Suppress SSL warnings in notebook mode (for cleaner development experience)
        if notebook_mode:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        cls._configured = True
    
    @classmethod
    def set_verbose(cls, verbose: bool = True):
        """Enable/disable verbose logging"""
        for logger in cls._loggers.values():
            for handler in logger.handlers:
                if isinstance(handler, RichHandler):
                    handler.setLevel(logging.DEBUG if verbose else logging.INFO)


def get_logger(name: str = "kepler") -> logging.Logger:
    """Convenience function to get a logger"""
    return KeplerLogger.get_logger(name)


def set_verbose(verbose: bool = True):
    """Convenience function to set verbose mode"""
    KeplerLogger.set_verbose(verbose)