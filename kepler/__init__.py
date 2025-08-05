"""
Kepler Framework - Simple ML for Industrial Data

A pragmatic framework for connecting Splunk data to machine learning models
and deploying them to Google Cloud Run.

Example usage:
    >>> import kepler as kp
    >>> data = kp.data.from_splunk("sensor_data", days=7)
    >>> model = kp.train.random_forest(data, target="anomaly")
    >>> predictions = model.predict(new_data)
    >>> kp.results.to_splunk(predictions, index="ml_predictions")
"""

__version__ = "0.1.0"
__author__ = "Kepler Team"

# Simple API modules for data scientists
from kepler import data
from kepler import train
from kepler import results

# Internal API (for advanced users) - moved to avoid circular imports
# from kepler.core.config import KeplerConfig, load_config
# from kepler.core.project import KeplerProject

__all__ = [
    "data",      # Simple data extraction: kp.data.from_splunk()
    "train",     # Simple training: kp.train.random_forest()
    "results",   # Simple results: kp.results.to_splunk()
    "__version__",
]