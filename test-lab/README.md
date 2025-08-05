# sensor-anomaly-detection

Kepler ML project for industrial data analysis.

## Quick Start

1. **Configure your environment:**
   ```bash
   cp .env.template .env
   # Edit .env with your actual credentials
   ```

2. **Validate prerequisites:**
   ```bash
   kepler validate
   ```

3. **Extract data from Splunk:**
   ```bash
   kepler extract "index=your_index | head 1000"
   ```

4. **Train a model:**
   ```bash
   kepler train data.csv --target your_target_column
   ```

5. **Deploy to Cloud Run:**
   ```bash
   kepler deploy model.pkl
   ```

## Project Structure

```
sensor-anomaly-detection/
├── kepler.yml          # Configuration file
├── data/
│   ├── raw/           # Raw data from Splunk
│   └── processed/     # Processed data for training
├── models/            # Trained model files
├── notebooks/         # Jupyter notebooks for exploration
├── scripts/           # Custom scripts
└── logs/              # Application logs
```

## Configuration

### 📁 Archivo de Proyecto: `kepler.yml`
Configura settings específicos del proyecto (SÍ se sube a Git):
- **Splunk**: URI del servidor, SSL, timeouts, índices
- **GCP**: Región, zona, configuración de despliegue  
- **ML**: Algoritmos, parámetros de entrenamiento
- **Deployment**: CPU, memoria, puertos

### 🔐 Variables de Entorno: `.env`
Configura datos sensibles (NO se sube a Git):
1. Copia `.env.template` a `.env`
2. Configura tus tokens y credenciales reales:
   - `SPLUNK_TOKEN`: Token de autenticación REST API
   - `SPLUNK_HEC_TOKEN`: Token del HTTP Event Collector  
   - `GCP_PROJECT_ID`: ID de tu proyecto Google Cloud

### 🌐 Flexibilidad de Configuración
- **URI de Splunk**: Se puede configurar en `kepler.yml` por proyecto
- **Tokens**: Se configuran en `.env` local (mayor seguridad)
- **Global vs Proyecto**: El proyecto tiene prioridad sobre configuración global

## Next Steps

1. Configure your Splunk and GCP credentials
2. Run `kepler validate` to check prerequisites
3. Start with data extraction: `kepler extract --help`

For more information, see the [Kepler documentation](https://github.com/company/kepler-framework).
