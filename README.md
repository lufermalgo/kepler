# Kepler Framework

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **Simple framework for industrial machine learning with Splunk and Google Cloud.**

## 🎯 What is Kepler?

Kepler is a pragmatic framework that connects industrial data from Splunk to machine learning models. **Currently validated and working:**

### ✅ **PRODUCTION READY FEATURES:**
- 📊 **Extract data from Splunk** with custom SPL queries (events & metrics)
- 🕐 **Time range control** for historical data analysis (`earliest`/`latest`)
- 🔧 **CLI and SDK** - use as command line tool OR `import kepler as kp`
- 🛡️ **Secure configuration** - credentials outside git, automatic validation
- 📈 **Index management** - auto-create/validate Splunk indexes
- 📊 **Jupyter integration** - clean notebooks for data scientists
- ⚡ **Real-time error handling** - Splunk errors captured and displayed clearly

### 🚧 **IN DEVELOPMENT:**
- 🤖 **Train ML models** (sklearn, XGBoost) - next sprint
- ☁️ **Deploy to Cloud Run** - planned
- 🔄 **Write predictions back to Splunk** - planned

---

## 📚 Documentación

### 📖 **Guías Especializadas**
- **[CLI Guide](./docs/CLI_GUIDE.md)** - Comandos de línea, automatización y DevOps
- **[SDK Guide](./docs/SDK_GUIDE.md)** - API Python para análisis de datos y notebooks
- **[Estado de Validación](./docs/VALIDATION_STATUS.md)** - Funcionalidades validadas con datos reales
- **[Índice Completo de Documentación](./docs/README.md)** - Navegación por toda la documentación

### 🎯 **Acceso Rápido por Rol**
- **👨‍💻 Científico de Datos:** [SDK Python](./docs/SDK_GUIDE.md#api-de-extracción-de-datos) | [Notebooks](./test-lab/notebooks/)
- **🔧 DevOps/Ingenieros:** [CLI Commands](./docs/CLI_GUIDE.md#comandos-principales) | [Automatización](./docs/CLI_GUIDE.md#automatización-y-scripts)
- **👔 Managers:** [Funcionalidades actuales](#production-ready-features) | [Estado validación](./docs/VALIDATION_STATUS.md)

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

**🎯 VALIDATED AND WORKING:**

```bash
# 1. Validate your setup
kepler validate  # ✅ Tests Splunk, GCP, indexes, connectivity

# 2. Extract events data
kepler extract "search index=kepler_lab sensor_type=temperature earliest=-7d" --output events.csv

# 3. Extract metrics data  
kepler extract "| mstats avg(_value) WHERE index=kepler_metrics metric_name=* earliest=-30d span=1h by metric_name" --output metrics.csv
```

**🐍 OR use as Python SDK in Jupyter:**

```python
import kepler as kp

# Extract data with time ranges
events = kp.data.from_splunk(
    spl="search index=kepler_lab sensor_type=temperature", 
    earliest="-7d", latest="now"
)

# Custom SPL for metrics
metrics = kp.data.from_splunk(
    spl="| mstats latest(_value) WHERE index=kepler_metrics metric_name=* earliest=-30d by metric_name"
)
```

**🚧 NEXT STEPS (in development):**
```bash
# Train models (next sprint)
kepler train events.csv --target temperature --algorithm random_forest

# Deploy to production (planned)
kepler deploy model.pkl
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

## 🔧 Estado del Proyecto

**Versión Actual: 0.1.0**

Ver detalles completos del estado en:
- **[Estado de Validación](./docs/VALIDATION_STATUS.md)** - Funcionalidades probadas con datos reales
- **[Roadmap Completo](./docs/SDK_CLI_GUIDE.md#evolución-y-roadmap)** - Evolución por sprints

## ✅ **Status de Validación**

**Datos reales probados:** 2,890 eventos + 16 métricas extraídos exitosamente

Para detalles completos de validación, ver **[Estado de Validación](./docs/VALIDATION_STATUS.md)**

### 🎯 **Próximos Pasos**
1. **🤖 Entrenamiento de Modelos** - Usando los datos validados
2. **☁️ Deployment a GCP Cloud Run** - Infraestructura configurada  
3. **🔄 Predicciones en Producción** - Escritura de resultados a Splunk

---

## 📦 **Instalación**

### Método Actual (GitHub)
```bash
git clone https://github.com/lufermalgo/kepler.git /tmp/kepler-install
cd /tmp/kepler-install && pip install .
rm -rf /tmp/kepler-install
```

### Futuro (PyPI - Sprint 13)
```bash
pip install kepler-framework  # Objetivo: instalación simple
```

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