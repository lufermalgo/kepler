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

# NEW: Unlimited library support API
from kepler import libs

# NEW: Unified training API (Task 1.8)
from kepler import train_unified

# NEW: AutoML capabilities (Task 1.11)
from kepler import automl

# NEW: Versioning and reproducibility (Task 5.1-5.6)
from kepler import versioning
from kepler import reproduce

# NEW: Model deployment (Task 6.2-6.5)
from kepler import deploy

# Internal API (for advanced users) - moved to avoid circular imports
# from kepler.core.config import KeplerConfig, load_config
# from kepler.core.project import KeplerProject

__all__ = [
    "data",          # Simple data extraction: kp.data.from_splunk()
    "train",         # Simple training: kp.train.random_forest()
    "train_unified", # Unified training API: kp.train_unified.train()
    "automl",        # AutoML capabilities: kp.automl.auto_train(), kp.automl.select_algorithm()
    "versioning",    # Data versioning and reproducibility: kp.versioning.version_data(), kp.versioning.create_unified_version()
    "reproduce",     # Reproduction system: kp.reproduce.from_version()
    "deploy",        # Model deployment: kp.deploy.to_cloud_run(), kp.deploy.validate()
    "results",       # Simple results: kp.results.to_splunk()
    "libs",          # Unlimited library support: kp.libs.install(), kp.libs.template()
    "__version__",
]