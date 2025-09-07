# Kepler Framework - Guía CLI (Command Line Interface)

> **Versión del Framework:** 0.1.0  
> **Última actualización:** Agosto 2025  
> **Estado:** Comandos Core Validados  
> **Audiencia:** DevOps, Ingenieros, Automatización

## 📋 Tabla de Contenidos

1. [Introducción al CLI](#introducción-al-cli)
2. [Instalación y Configuración](#instalación-y-configuración)
3. [Comandos Principales](#comandos-principales)
4. [Automatización y Scripts](#automatización-y-scripts)
5. [Troubleshooting](#troubleshooting)
6. [Evolución del CLI](#evolución-del-cli)

---

## 🎯 Introducción al CLI

El **CLI de Kepler** está diseñado para:

### **🖥️ Operaciones Rápidas**
- Validación de entorno y diagnósticos
- Extracción de datos desde línea de comandos
- Configuración y gestión de proyectos

### **🤖 Automatización**
- Scripts de DevOps y pipelines
- Integración en sistemas CI/CD
- Tareas programadas y cron jobs

### **🔧 Administración**
- Gestión de configuración global
- Validación de conectividad
- Diagnóstico de problemas

---

## ⚙️ Instalación y Configuración

### Instalación del Framework

```bash
# Crear entorno virtual (RECOMENDADO)
python -m venv kepler-env
source kepler-env/bin/activate  # Linux/macOS
# kepler-env\Scripts\activate   # Windows

# Instalar desde GitHub (método actual)
git clone https://github.com/lufermalgo/kepler.git /tmp/kepler-install
cd /tmp/kepler-install
pip install .
cd ~ && rm -rf /tmp/kepler-install

# Verificar instalación
kepler --version  # ✅ Debe mostrar: 0.1.0
```

### Configuración Inicial

```bash
# Crear estructura de configuración
kepler config init

# Esto crea: ~/.kepler/config.yml
```

### Configurar Credenciales

Editar el archivo `~/.kepler/config.yml`:

```yaml
# ~/.kepler/config.yml
splunk:
  host: "http://localhost:8089"  # Tu servidor Splunk
  hec_host: "http://localhost:8088"  # HTTP Event Collector
  token: "tu-token-rest-api"     # Token para REST API
  hec_token: "tu-token-hec"      # Token para HEC
  verify_ssl: false              # Para desarrollo local

gcp:
  project_id: "tu-proyecto-gcp"
  region: "us-central1"
  credentials_path: "~/.gcp/service-account.json"
```

### Validar Configuración

```bash
# Validación completa en 5 pasos
kepler validate

# Salida esperada:
# ✅ Step 1/5: Prerequisites validation
# ✅ Step 2/5: GCP configuration  
# ✅ Step 3/5: Project configuration
# ✅ Step 4/5: Splunk connectivity
# ✅ Step 5/5: Splunk indexes validation
```

---

## 🖥️ Comandos Principales

### **1. `kepler validate` - Diagnóstico Completo**

```bash
# Validación completa del entorno
kepler validate

# Validación específica de Splunk
kepler validate --splunk-only

# Validación con output detallado
kepler validate --verbose
```

**Qué valida:**
- ✅ Python 3.8+ instalado
- ✅ Dependencias del framework
- ✅ Configuración GCP
- ✅ Conectividad Splunk (REST API + HEC)
- ✅ Existencia de índices `kepler_lab` y `kepler_metrics`
- ✅ Auto-creación de índices si no existen

### **2. `kepler extract` - Extracción de Datos**

```bash
# Extracción básica con SPL personalizado
kepler extract "search index=kepler_lab" --output data.csv

# Extracción con rango de tiempo
kepler extract "search index=kepler_lab sensor_type=temperature" \
    --earliest "-7d" --latest "now" --output temperature_data.csv

# Extracción de métricas
kepler extract "| mstats avg(_value) WHERE index=kepler_metrics metric_name=* earliest=-30d span=1h by metric_name" \
    --output metrics_hourly.csv

# Extracción con límite de registros
kepler extract "search index=kepler_lab" --limit 1000 --output sample_data.csv
```

**Parámetros disponibles:**
- `--output`: Archivo de salida (CSV)
- `--earliest`: Tiempo inicial (ej: `-7d`, `-24h`, `-1h`)
- `--latest`: Tiempo final (por defecto: `now`)
- `--limit`: Máximo número de registros
- `--format`: Formato de salida (`csv`, `json`) - por defecto: `csv`

### **3. `kepler config` - Gestión de Configuración**

```bash
# Inicializar configuración
kepler config init

# Mostrar configuración actual (sin credenciales)
kepler config show

# Validar configuración
kepler config validate

# Mostrar ubicaciones de archivos
kepler config paths
```

### **4. Comandos de Ayuda**

```bash
# Ayuda general
kepler --help

# Ayuda de comando específico
kepler extract --help
kepler validate --help
kepler config --help
```

---

## 🤖 Automatización y Scripts

### **Script de Pipeline Diario**

```bash
#!/bin/bash
# extract_daily_data.sh - Pipeline automatizado

echo "=== PIPELINE DIARIO DE EXTRACCIÓN ==="

# 1. Validar entorno
echo "Validando configuración..."
kepler validate || exit 1

# 2. Extraer datos del día anterior
echo "Extrayendo eventos del día anterior..."
kepler extract "search index=kepler_lab earliest=-1d@d latest=@d" \
    --output "data/events_$(date +%Y%m%d).csv"

# 3. Extraer métricas horarias
echo "Extrayendo métricas horarias..."
kepler extract "| mstats avg(_value) WHERE index=kepler_metrics metric_name=* earliest=-1d@d latest=@d span=1h by metric_name" \
    --output "data/metrics_hourly_$(date +%Y%m%d).csv"

# 4. Generar reporte básico
echo "Generando reporte..."
python generate_daily_report.py "data/events_$(date +%Y%m%d).csv"

echo "Pipeline completado exitosamente"
```

### **Integración con Cron**

```bash
# Cron job para extracción diaria a las 2:00 AM
# 0 2 * * * /path/to/extract_daily_data.sh >> /var/log/kepler_daily.log 2>&1
```

### **Script de Monitoreo**

```bash
#!/bin/bash
# monitor_kepler.sh - Monitoreo continuo

while true; do
    echo "[$(date)] Checking Kepler connectivity..."
    
    if kepler validate --splunk-only > /dev/null 2>&1; then
        echo "✅ Splunk connectivity OK"
    else
        echo "❌ Splunk connectivity FAILED"
        # Enviar alerta (email, Slack, etc.)
        curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"⚠️ Kepler Splunk connectivity failed"}' \
            $SLACK_WEBHOOK_URL
    fi
    
    sleep 300  # Check every 5 minutes
done
```

---

## 🔍 Troubleshooting

### **Problemas Comunes**

#### **Error: Command not found**
```bash
# Verificar instalación
which kepler
# Si no aparece, reactivar entorno virtual
source kepler-env/bin/activate
```

#### **Error: Configuration file not found**
```bash
# Inicializar configuración
kepler config init
# Verificar ubicación
kepler config paths
```

#### **Error: Splunk connection failed**
```bash
# Validar conectividad específica
kepler validate --splunk-only --verbose
# Verificar tokens y URLs en ~/.kepler/config.yml
```

#### **Error: Permission denied**
```bash
# Verificar permisos del archivo de configuración
ls -la ~/.kepler/config.yml
# Debe ser 600 (solo el usuario puede leer/escribir)
chmod 600 ~/.kepler/config.yml
```

### **Logs y Diagnóstico**

```bash
# Ejecutar con verbosidad máxima
kepler --verbose extract "search index=kepler_lab"

# Ver configuración actual (sin credenciales)
kepler config show

# Verificar paths de configuración
kepler config paths
```

---

## 🚧 Comandos en Desarrollo

### **Sprint Actual - ML Training**
```bash
# Comandos de ML que se van a implementar
kepler train data.csv --target temperature --algorithm random_forest
kepler train data.csv --target anomaly --algorithm xgboost --params config.json
kepler model list  # Listar modelos entrenados
kepler model info model_20241215_rf.pkl  # Info del modelo
```

### **Sprint 11-12 - Deployment**
```bash
# Deploy a Cloud Run
kepler deploy model.pkl --name temp-predictor --env production

# Gestión de deployments  
kepler deploy list
kepler deploy logs temp-predictor --tail
kepler deploy scale temp-predictor --instances 3
```

### **Sprint 13-16 - Funcionalidades Avanzadas**
- 🔄 **Pipeline completos** de CI/CD
- 📊 **Dashboards automáticos** en Splunk
- 🔧 **Gestión avanzada** de modelos
- 📦 **Distribución PyPI** (`pip install kepler-framework`)

---

## 📚 Recursos Adicionales

### **Archivos de Configuración**
- `~/.kepler/config.yml` - Configuración global segura
- `proyecto/kepler.yml` - Configuración específica del proyecto

### **Ejemplos de Uso**
- Ver **[casos de uso completos](./SDK_GUIDE.md#casos-de-uso-completos)** en la guía del SDK
- Scripts de ejemplo en `test-lab/scripts/`

### **Documentación Relacionada**
- **[SDK Guide](./SDK_GUIDE.md)** - API Python para notebooks
- **[Estado de Validación](./VALIDATION_STATUS.md)** - Funcionalidades probadas

---

> **💡 Tip para DevOps:** Comienza siempre con `kepler validate` en tus scripts para asegurar que el entorno esté correctamente configurado antes de ejecutar operaciones de datos.

> **🎯 Estado Actual:** Todos los comandos de extracción y validación están completamente funcionales y listos para uso en producción.