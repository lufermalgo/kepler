# Kepler Framework

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **Simple framework for industrial machine learning with Splunk and Google Cloud.**

## 🎯 What is Kepler?

Kepler is a pragmatic framework that connects industrial data from Splunk to machine learning models deployed on Google Cloud Run. It's designed for data scientists who want to:

- 📊 **Extract data from Splunk** with simple SPL queries
- 🤖 **Train ML models** (sklearn, XGBoost) without DevOps complexity  
- ☁️ **Deploy models to production** with a single command
- 🔄 **Write predictions back to Splunk** automatically via HEC
- 🛡️ **Manage credentials securely** outside your project repository

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** installed on your system
- **Splunk Enterprise** with REST API access
- **Google Cloud Project** with Cloud Run enabled

### 1. Setup Python Environment

**⚠️ Always use a virtual environment in your WORK directory (not in the cloned repo):**

```bash
# Navigate to your work directory
cd ~/my-projects  # Or wherever you want to work

# Create virtual environment
python -m venv kepler-env

# Activate it
# On macOS/Linux:
source kepler-env/bin/activate
# On Windows:
# kepler-env\Scripts\activate

# Verify Python version
python --version  # Should be 3.8+
```

### 2. Install Kepler Framework

**From GitHub (current):**
```bash
# Download and install (in a temporary directory)
git clone https://github.com/lufermalgo/kepler.git /tmp/kepler-install
cd /tmp/kepler-install
pip install .  # Note: no "-e" for end users

# Clean up - you can now delete the cloned repository
cd ~/my-projects  # Back to your work directory
rm -rf /tmp/kepler-install  # ✅ Safe to delete after installation

# Verify installation
kepler --help
```

**From PyPI (coming soon - target: Sprint 13):**
```bash
pip install kepler-framework  # Much simpler!
```

### 3. Initial Configuration

**Create secure global configuration:**
```bash
kepler config init
```

**Edit your credentials (one-time setup):**
```bash
# Edit the secure config file
nano ~/.kepler/config.yml
```

### 4. Create Your First Project

```bash
kepler init my-industrial-project
cd my-industrial-project
```

### 5. Verify Setup

```bash
kepler config validate
```

### 6. Basic Workflow

```bash
# 1. Extract data from Splunk
kepler extract "search index=sensors | head 1000" --output sensor_data.csv

# 2. Train a model
kepler train sensor_data.csv --target temperature --algorithm random_forest

# 3. Deploy to Cloud Run (coming in Sprint 9-10)
kepler deploy model_random_forest_*.pkl

# 4. Make predictions (coming in Sprint 9-10)  
kepler predict https://your-api-url.run.app/predict '{"pressure": 2.5, "flow": 100}'
```

## 💡 Best Practices

### Environment Management
```bash
# ✅ DO: Create virtual environment in your WORK directory
cd ~/my-projects
python -m venv kepler-env
source kepler-env/bin/activate  # Linux/macOS
# kepler-env\Scripts\activate   # Windows

# ✅ DO: Install from GitHub (current method)
git clone https://github.com/lufermalgo/kepler.git /tmp/kepler-install
pip install /tmp/kepler-install
rm -rf /tmp/kepler-install

# ❌ DON'T: Use -e flag unless you're developing Kepler itself
pip install -e .  # Only for contributors
```

### Security
```bash
# ✅ DO: Use global configuration for credentials
kepler config init
nano ~/.kepler/config.yml

# ❌ DON'T: Put credentials in your project files
# Never commit sensitive data to git
```

### Data Management
```bash
# ✅ DO: Start with small data samples
kepler extract "search index=sensors | head 1000"

# ✅ DO: Validate data quality before training
kepler train data.csv --target temperature  # Includes automatic validation

# ✅ DO: Use descriptive target column names
kepler train sensor_data.csv --target gas_consumption_m3h
```

## 📋 Prerequisites

- Python 3.8+
- Splunk Enterprise (with REST API access)
- Google Cloud Project (with Cloud Run enabled)
- Required Python packages (auto-installed):
  - pandas, numpy, scikit-learn, xgboost
  - splunk-sdk, google-cloud-run
  - typer, pydantic, rich

## 🏗️ Project Structure

When you run `kepler init`, it creates:

```
my-project/
├── kepler.yml          # Configuration file
├── .env.template       # Environment variables template
├── data/
│   ├── raw/           # Raw data from Splunk
│   └── processed/     # Processed data for training
├── models/            # Trained model files (.pkl)
├── notebooks/         # Jupyter notebooks for exploration
└── logs/              # Application logs
```

## 🔧 Configuration

Kepler uses **two configuration files** for security and flexibility:

### Global Configuration (Sensitive Data)

**File:** `~/.kepler/config.yml` *(secure, outside your project)*

```yaml
# Global configuration with sensitive credentials
splunk:
  host: "https://your-splunk-server:8089"
  token: "your-splunk-auth-token"
  hec_token: "your-splunk-hec-token"
  verify_ssl: true
  metrics_index: "kepler_metrics"

gcp:
  project_id: "your-gcp-project-id"
  region: "us-central1"
  service_account_file: "/path/to/service-account.json"

mlflow:
  tracking_uri: "http://your-mlflow-server:5000"
```

### Project Configuration (Public Settings)

**File:** `your-project/kepler.yml` *(safe to commit to git)*

```yaml
project_name: my-industrial-project

training:
  default_algorithm: "random_forest"
  test_size: 0.2
  random_state: 42

deployment:
  service_name: "my-project-api"
  port: 8080
```

### Configuration Commands

```bash
# Initialize global config (one-time setup)
kepler config init

# View current configuration (sanitized)
kepler config show

# Validate credentials and connectivity
kepler config validate

# Show config file location
kepler config path
```

## 🎯 Real-World Use Cases

### Predictive Maintenance (Available Now)
```bash
# Extract hourly sensor data for industrial pumps
kepler extract "search index=sensors asset_id=PUMP_001 | stats avg(vibration_mm_s) as vibration, avg(temperature_C) as temp, avg(pressure_bar) as pressure by _time span=1h" --output pump_data.csv

# Train failure prediction model
kepler train pump_data.csv --target needs_maintenance --algorithm random_forest

# Model automatically saved as: model_random_forest_YYYYMMDD_HHMMSS.pkl
```

### Gas Consumption Optimization (Available Now)
```bash
# Extract process data by batch
kepler extract "search index=process_data | stats avg(gas_consumption_m3h) as gas_usage, avg(temperature_C) as temp, avg(pressure_bar) as pressure, avg(flow_rate_m3h) as flow by batch_id" --output process_data.csv

# Train optimization model
kepler train process_data.csv --target gas_usage --algorithm xgboost

# Results include performance metrics and feature importance
```

### Quality Control Classification (Available Now)
```bash
# Extract sensor data for quality assessment
kepler extract "search index=sensors | stats avg(sensor_*) by batch_id | eval quality_ok=if(sensor_variance<0.1, 1, 0)" --output quality_data.csv

# Train quality classifier
kepler train quality_data.csv --target quality_ok --test-size 0.3

# Get classification metrics: accuracy, precision, recall, F1-score
```

### Coming Soon (Sprint 9-10)
- **Automatic deployment** to Google Cloud Run
- **Real-time predictions** via REST API
- **Automatic result writing** back to Splunk HEC

## 📊 Supported ML Algorithms

**Current (MVP):**
- Random Forest (classification/regression)
- Linear Regression
- XGBoost (classification/regression)

**Coming Soon:**
- More sklearn algorithms
- Deep learning (PyTorch/TensorFlow)
- Time series forecasting
- Custom model support

## 🔄 Data Flow

```
[Splunk Data] → [kepler extract] → [CSV Files] → [kepler train] → [Model.pkl] → [kepler deploy] → [Cloud Run API] → [Predictions] → [Splunk HEC]
```

## 🛠️ Development Status

**Current Version: 0.1.0 (MVP Ready)**

✅ **Completed Features:**
- ✅ **CLI Foundation:** Complete CLI with typer, rich output, error handling
- ✅ **Project Management:** `kepler init`, `kepler config` commands
- ✅ **Secure Configuration:** Global config management (`~/.kepler/config.yml`)
- ✅ **Splunk Integration:** Bidirectional data flow (extract + HEC write)
- ✅ **ML Training:** sklearn RandomForest, Linear models, XGBoost basic
- ✅ **Data Validation:** Quality assessment, cleaning, ML readiness checks
- ✅ **Testing Suite:** Unit, integration, and realistic end-to-end tests
- ✅ **Connection Resilience:** SSL fallbacks, error handling, connectivity validation

🚧 **In Progress (Sprint 6):**
- XGBoost enhancement and hyperparameter tuning
- Model evaluation and reporting improvements
- Advanced model serialization with metadata

📋 **Next Priorities:**
- Sprint 7-8: Advanced ML features (time series, model comparison)
- Sprint 9-10: Google Cloud Run deployment (`kepler deploy`)
- Sprint 11-12: Production monitoring and model versioning
- Sprint 13-16: Comprehensive documentation and user validation

**🎯 MVP Status:** Ready for early adopter testing with core Splunk ↔ ML workflow.

## 📖 Documentation Roadmap

### Current Documentation (MVP)
- ✅ **README.md** - Quick start and basic usage
- ✅ **Inline Help** - `kepler --help`, `kepler <command> --help`
- ✅ **Configuration Guide** - Global and project configuration
- ✅ **Integration Tests** - `/tests/integration/README.md`

### Planned Documentation (Post-MVP)
- 📋 **Complete CLI Reference** - Detailed documentation for all commands and options
- 📋 **SDK Documentation** - Full API reference for `import kepler` usage in notebooks
- 📋 **Tutorials & Examples** - Step-by-step guides for common industrial use cases
- 📋 **Architecture Guide** - Technical deep-dive for contributors and advanced users
- 📋 **Deployment Guide** - Production deployment best practices
- 📋 **Troubleshooting Guide** - Common issues and solutions

> 💡 **Note:** Comprehensive CLI and SDK documentation will be developed as part of the roadmap in Sprints 13-16, based on user feedback and real-world usage patterns.

## 📦 **PyPI Publishing Roadmap**

### What is PyPI?
[PyPI (Python Package Index)](https://pypi.org/) is the official repository for Python packages. When a package is published to PyPI, users can install it with simple `pip install package-name` commands.

### Current Status: GitHub Installation
- **Now:** `git clone` + `pip install .` (temporary solution)
- **Target:** Sprint 13 - PyPI publishing

### PyPI Publishing Process (Sprint 13)
```bash
# What will be done internally:
1. Create PyPI account
2. Configure package metadata (already in pyproject.toml)
3. Build package: python -m build
4. Upload to PyPI: twine upload dist/*
5. Test installation: pip install kepler-framework
```

### Benefits After PyPI Publishing
- ✅ **Simple installation:** `pip install kepler-framework`
- ✅ **Automatic updates:** `pip install --upgrade kepler-framework`
- ✅ **Version management:** `pip install kepler-framework==0.2.0`
- ✅ **No repository cloning** required
- ✅ **Global accessibility** for data scientists worldwide

## 🤝 Contributing

This project is currently in active development. External contributions will be welcomed after MVP completion and initial documentation is finalized.

**For now:**
- 🐛 **Report Issues:** [GitHub Issues](https://github.com/lufermalgo/kepler/issues)
- 💡 **Feature Requests:** Contact development team
- 📖 **Documentation:** Help us improve this README

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support

- 📚 **Documentation:** This README + `kepler --help`
- 🐛 **Issues:** [GitHub Issues](https://github.com/lufermalgo/kepler/issues)
- 📧 **Contact:** Development team for enterprise usage
- 🚀 **Status:** Early adopter phase - feedback welcome!

---

**Kepler Framework** - Making industrial ML accessible to data scientists. 🚀