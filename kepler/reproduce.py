"""
Kepler Reproduction API - Task 5.6 Implementation
System for reproducing any version of the project

Provides comprehensive reproduction capabilities for:
- Unified version reproduction (Git + DVC + MLflow)
- Data version reproduction
- Pipeline version reproduction
- Experiment reproduction
- Model reproduction
"""

from kepler.versioning import (
    reproduce_from_version,
    get_reproduction_summary,
    ReproductionResult
)

# Re-export the main reproduction function
from_version = reproduce_from_version

__all__ = [
    "from_version",
    "reproduce_from_version", 
    "get_reproduction_summary",
    "ReproductionResult"
]
