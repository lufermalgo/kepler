"""
Custom exceptions for Kepler framework

Provides specific exception types for better error handling and user experience.
"""

import sys
import traceback
from typing import Optional


class KeplerError(Exception):
    """Base exception for all Kepler-related errors"""
    
    def __init__(self, message: str, details: Optional[str] = None, suggestion: Optional[str] = None):
        self.message = message
        self.details = details
        self.suggestion = suggestion
        super().__init__(self.message)
    
    def get_user_message(self, verbose: bool = False) -> str:
        """Get a user-friendly error message"""
        msg = f"❌ {self.message}"
        
        if self.details:
            msg += f"\n   Details: {self.details}"
        
        if self.suggestion:
            msg += f"\n   💡 Suggestion: {self.suggestion}"
        
        if verbose:
            msg += f"\n   Stack trace:\n{traceback.format_exc()}"
        
        return msg


class ConfigurationError(KeplerError):
    """Raised when there's an issue with configuration"""
    
    def __init__(self, message: str, config_file: Optional[str] = None, suggestion: Optional[str] = None):
        details = f"Configuration file: {config_file}" if config_file else None
        super().__init__(message, details, suggestion)


class SplunkConnectionError(KeplerError):
    """Raised when Splunk connection fails"""
    
    def __init__(self, message: str, splunk_host: Optional[str] = None, suggestion: Optional[str] = None):
        details = f"Splunk host: {splunk_host}" if splunk_host else None
        default_suggestion = "Check your Splunk credentials and network connectivity"
        super().__init__(message, details, suggestion or default_suggestion)


class DataExtractionError(KeplerError):
    """Raised when data extraction from Splunk fails"""
    
    def __init__(self, message: str, query: Optional[str] = None, suggestion: Optional[str] = None):
        details = f"Query: {query}" if query else None
        default_suggestion = "Verify your SPL query syntax and data availability"
        super().__init__(message, details, suggestion or default_suggestion)


class ModelTrainingError(KeplerError):
    """Raised when model training fails"""
    
    def __init__(self, message: str, data_info: Optional[str] = None, suggestion: Optional[str] = None):
        details = f"Data info: {data_info}" if data_info else None
        default_suggestion = "Check your data quality and target column"
        super().__init__(message, details, suggestion or default_suggestion)


class DeploymentError(KeplerError):
    """Raised when model deployment fails"""
    
    def __init__(self, message: str, service_name: Optional[str] = None, suggestion: Optional[str] = None):
        details = f"Service: {service_name}" if service_name else None
        default_suggestion = "Check your GCP credentials and project configuration"
        super().__init__(message, details, suggestion or default_suggestion)


class ValidationError(KeplerError):
    """Raised when validation fails"""
    
    def __init__(self, message: str, component: Optional[str] = None, suggestion: Optional[str] = None):
        details = f"Component: {component}" if component else None
        default_suggestion = "Run 'kepler validate' to check prerequisites"
        super().__init__(message, details, suggestion or default_suggestion)


def handle_exception(exc: Exception, verbose: bool = False) -> int:
    """
    Handle exceptions in a user-friendly way
    
    Returns appropriate exit code
    """
    from kepler.utils.logging import get_logger
    logger = get_logger()
    
    if isinstance(exc, KeplerError):
        # Custom Kepler errors - show user-friendly message
        print(exc.get_user_message(verbose=verbose))
        logger.error(f"KeplerError: {exc.message}", exc_info=verbose)
        return 1
    elif isinstance(exc, KeyboardInterrupt):
        # User cancelled operation
        print("\n⚠️  Operation cancelled by user")
        logger.info("Operation cancelled by user (KeyboardInterrupt)")
        return 130  # Standard exit code for SIGINT
    elif isinstance(exc, FileNotFoundError):
        # File not found - common issue
        print(f"❌ File not found: {exc.filename}")
        print("💡 Suggestion: Check the file path and ensure the file exists")
        logger.error(f"FileNotFoundError: {exc}", exc_info=verbose)
        return 2
    elif isinstance(exc, PermissionError):
        # Permission denied - common issue
        print(f"❌ Permission denied: {exc.filename}")
        print("💡 Suggestion: Check file permissions or run with appropriate privileges")
        logger.error(f"PermissionError: {exc}", exc_info=verbose)
        return 13
    else:
        # Unexpected errors
        if verbose:
            print(f"❌ Unexpected error: {exc}")
            print(f"Stack trace:\n{traceback.format_exc()}")
        else:
            print(f"❌ Unexpected error: {exc}")
            print("💡 Run with --verbose for more details")
        
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        return 1


def setup_exception_handler():
    """Setup global exception handler for uncaught exceptions"""
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Let KeyboardInterrupt be handled normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Handle other exceptions
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        exit_code = handle_exception(exc_value, verbose=verbose)
        sys.exit(exit_code)
    
    sys.excepthook = exception_handler