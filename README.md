# Kepler Framework

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simple framework for industrial machine learning with Splunk and Google Cloud.

## 🎯 What is Kepler?

Kepler is a pragmatic framework that connects industrial data from Splunk to machine learning models deployed on Google Cloud Run. It's designed for data scientists who want to:

- Extract data from Splunk easily
- Train ML models (sklearn, XGBoost) without DevOps complexity
- Deploy models to production with a single command
- Write predictions back to Splunk automatically

## 🚀 Quick Start

### Installation

```bash
pip install kepler-framework
```

### Initialize a Project

```bash
kepler init my-industrial-project
cd my-industrial-project
```

### Configure Credentials

```bash
cp .env.template .env
# Edit .env with your Splunk and GCP credentials
```

### Validate Setup

```bash
kepler validate
```

### Basic Workflow

```bash
# 1. Extract data from Splunk
kepler extract "index=sensors | head 1000" --output sensor_data.csv

# 2. Train a model
kepler train sensor_data.csv --target temperature

# 3. Deploy to Cloud Run
kepler deploy model.pkl

# 4. Make predictions
kepler predict https://your-api-url.run.app/predict '{"pressure": 2.5, "flow": 100}'
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

Edit `kepler.yml` to configure:

```yaml
project_name: my-industrial-project

splunk:
  host: "https://localhost:8089"
  token: "${SPLUNK_TOKEN}"
  hec_token: "${SPLUNK_HEC_TOKEN}"
  metrics_index: "kepler_metrics"

gcp:
  project_id: "${GCP_PROJECT_ID}"
  region: "us-central1"

training:
  default_algorithm: "random_forest"
  test_size: 0.2
  random_state: 42

deployment:
  service_name: "my-project-api"
  port: 8080
  cpu: "1"
  memory: "2Gi"
```

## 🎯 Use Cases

### Predictive Maintenance
```bash
kepler extract "index=sensors asset_id=PUMP_001 | stats avg(vibration), avg(temperature) by _time span=1h"
kepler train pump_data.csv --target failure_risk
kepler deploy pump_failure_model.pkl
```

### Process Optimization
```bash
kepler extract "index=process_data | stats avg(gas_consumption), avg(temperature), avg(pressure) by batch_id"
kepler train process_data.csv --target gas_consumption --algorithm xgboost
kepler deploy gas_optimization_model.pkl
```

### Anomaly Detection
```bash
kepler extract "index=sensors | stats avg(sensor_*) by _time span=5m"
kepler train sensor_data.csv --target is_anomaly
kepler deploy anomaly_detector.pkl
```

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

**Current Version: 0.1.0 (MVP)**

✅ **Completed:**
- Project initialization (`kepler init`)
- Configuration management
- Prerequisites validation (`kepler validate`)

🚧 **In Progress (Sprint 1-2):**
- CLI foundation and error handling
- Testing framework setup

📋 **Upcoming Sprints:**
- Sprint 3-4: Splunk integration (`kepler extract`)
- Sprint 5-6: ML training (`kepler train`)
- Sprint 7-8: Training enhancements
- Sprint 9-10: Cloud deployment (`kepler deploy`)
- Sprint 11-12: Deployment enhancements
- Sprint 13-16: UX, documentation, validation

## 🤝 Contributing

This is currently an internal project. External contributions will be welcomed after MVP completion.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 📞 Support

- Documentation: Coming soon
- Issues: Contact the development team
- Slack: #kepler-framework (internal)

---

**Kepler Framework** - Making industrial ML accessible to data scientists. 🚀