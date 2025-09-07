"""
Custom exceptions for Kepler framework

Provides specific exception types for better error handling and user experience.
"""

import sys
import traceback
from typing import Optional, Dict


class KeplerError(Exception):
    """Base exception for all Kepler-related errors with standardized structure"""
    
    def __init__(self, code: str, message: str, hint: str = None, 
                 context: Dict = None, retryable: bool = False):
        self.code = code          # Código estandarizado
        self.message = message    # Mensaje claro
        self.hint = hint         # Sugerencia de remediación
        self.context = context or {}  # Contexto adicional
        self.retryable = retryable    # ¿Se puede reintentar?
        super().__init__(self.message)
    
    def get_user_message(self, verbose: bool = False) -> str:
        """Get a user-friendly error message"""
        msg = f"❌ [{self.code}] {self.message}"
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            msg += f"\n   Context: {context_str}"
        
        if self.hint:
            msg += f"\n   💡 Hint: {self.hint}"
            
        if self.retryable:
            msg += f"\n   🔄 This operation can be retried"
        
        if verbose:
            msg += f"\n   Stack trace:\n{traceback.format_exc()}"
        
        return msg


class ConfigurationError(KeplerError):
    """Raised when there's an issue with configuration"""
    
    def __init__(self, message: str, config_file: Optional[str] = None, hint: Optional[str] = None):
        context = {"config_file": config_file} if config_file else {}
        super().__init__(
            code="CONFIG_001",
            message=message,
            hint=hint or "Check configuration file syntax and required fields",
            context=context,
            retryable=False
        )


class SplunkConnectionError(KeplerError):
    """Raised when connection to Splunk fails"""
    
    def __init__(self, message: str, splunk_host: Optional[str] = None, hint: Optional[str] = None):
        context = {"host": splunk_host} if splunk_host else {}
        super().__init__(
            code="SPLUNK_001",
            message=message,
            hint=hint or "Check Splunk host URL and credentials",
            context=context,
            retryable=True
        )


class DataExtractionError(KeplerError):
    """Raised when data extraction from Splunk fails"""
    
    def __init__(self, message: str, query: Optional[str] = None, hint: Optional[str] = None):
        context = {"query": query} if query else {}
        super().__init__(
            code="DATA_001",
            message=message,
            hint=hint or "Check your SPL query syntax and index permissions",
            context=context,
            retryable=True
        )


class ModelTrainingError(KeplerError):
    """Raised when model training fails"""
    
    def __init__(self, message: str, algorithm: Optional[str] = None, hint: Optional[str] = None):
        context = {"algorithm": algorithm} if algorithm else {}
        super().__init__(
            code="TRAINING_001",
            message=message,
            hint=hint or "Check your data quality and target column",
            context=context,
            retryable=True
        )


class DeploymentError(KeplerError):
    """Raised when model deployment fails"""
    
    def __init__(self, message: str, service_name: Optional[str] = None, hint: Optional[str] = None):
        context = {"service": service_name} if service_name else {}
        super().__init__(
            code="DEPLOY_001",
            message=message,
            hint=hint or "Check cloud credentials and service configuration",
            context=context,
            retryable=True
        )


class ValidationError(KeplerError):
    """Raised when validation fails"""
    
    def __init__(self, message: str, component: Optional[str] = None, hint: Optional[str] = None):
        context = {"component": component} if component else {}
        super().__init__(
            code="VALIDATE_001",
            message=message,
            hint=hint or "Run 'kepler validate' to check prerequisites",
            context=context,
            retryable=True
        )


class LibraryManagementError(KeplerError):
    """Raised when library installation or management fails"""
    
    def __init__(self, message: str, library_name: Optional[str] = None, hint: Optional[str] = None):
        context = {"library": library_name} if library_name else {}
        super().__init__(
            code="LIBRARY_001",
            message=message,
            hint=hint or "Check library name, version, and source URL",
            context=context,
            retryable=True
        )


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