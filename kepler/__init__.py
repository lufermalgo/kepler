"""
Kepler Framework - Simple ML for Industrial Data

A pragmatic framework for connecting Splunk data to machine learning models
and deploying them to Google Cloud Run.
"""

__version__ = "0.1.0"
__author__ = "Kepler Team"

# Public API exports for SDK usage
from kepler.core.config import KeplerConfig
from kepler.core.project import KeplerProject

__all__ = [
    "KeplerConfig",
    "KeplerProject",
    "__version__",
]