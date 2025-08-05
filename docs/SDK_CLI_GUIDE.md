# Kepler Framework - Guía Completa SDK y CLI

> **Versión del Framework:** 0.1.0  
> **Última actualización:** Diciembre 2024  
> **Estado:** Funcionalidades Core Validadas

## 📋 Tabla de Contenidos

1. [Introducción y Filosofía](#introducción-y-filosofía)
2. [Instalación y Configuración](#instalación-y-configuración)
3. [CLI - Línea de Comandos](#cli---línea-de-comandos)
4. [SDK - Python API](#sdk---python-api)
5. [Casos de Uso Completos](#casos-de-uso-completos)
6. [Evolución y Roadmap](#evolución-y-roadmap)

---

## 🎯 Introducción y Filosofía

Kepler Framework ofrece **dos formas de interactuar** con la misma funcionalidad:

### **🖥️ CLI (Command Line Interface)**
- Para **operaciones rápidas y scripts**
- Ideal para **DevOps y automatización**
- **Validación de entorno** y diagnósticos

### **🐍 SDK (Software Development Kit)**
- Para **análisis interactivo en Jupyter**
- Ideal para **científicos de datos**
- **Integración en notebooks** y scripts Python

> **Principio Clave:** Una sola lógica, dos interfaces. Todo lo que hace el CLI también lo puede hacer el SDK.

---

## ⚙️ Instalación y Configuración COMPLETA

### Paso 1: Instalación del Framework

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

### Paso 2: Configuración Inicial

```bash
# Crear estructura de configuración
kepler config init

# Esto crea: ~/.kepler/config.yml
```

### Paso 3: Configurar Credenciales

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

### Paso 4: Validar Configuración

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

## 🖥️ CLI - Línea de Comandos

### **Comandos Principales (Validados y Funcionando)**

#### **1. `kepler validate` - Diagnóstico Completo**

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

#### **2. `kepler extract` - Extracción de Datos**

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

#### **3. `kepler config` - Gestión de Configuración**

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

### **Comandos en Desarrollo (Próximos Sprints)**

```bash
# 🚧 EN DESARROLLO - Sprint actual
kepler train data.csv --target temperature --algorithm random_forest
kepler train data.csv --target anomaly --algorithm xgboost --test-size 0.3

# 🚧 PLANEADO - Sprint 9-10  
kepler deploy model.pkl --name my-model --env production
kepler predict https://endpoint.run.app/predict '{"temperature": 25.5}'

# 🚧 PLANEADO - Sprint 11-12
kepler monitor --model my-model --dashboard splunk
kepler logs --model my-model --tail
```

---

## 🐍 SDK - Python API

### **Import y Configuración**

```python
# Import súper simple - sin configuración manual
import kepler as kp

# El SDK usa automáticamente la configuración de ~/.kepler/config.yml
print(f"Kepler Framework version: {kp.__version__}")
```

### **Extracción de Datos con `kp.data.from_splunk()`**

#### **Extracción Básica de Eventos**

```python
# Eventos de sensores - últimos 7 días
eventos = kp.data.from_splunk(
    spl="search index=kepler_lab sensor_type=temperature",
    earliest="-7d",
    latest="now"
)

print(f"Eventos extraídos: {len(eventos)}")
print(f"Columnas: {list(eventos.columns)}")
print(eventos.head())
```

**Resultado validado:**
```
Eventos extraídos: 867
Columnas: ['_time', 'sensor_id', 'sensor_type', 'area', 'value', 'unit']
```

#### **Extracción de Métricas**

```python
# Métricas - últimos 30 días  
metricas = kp.data.from_splunk(
    spl="| mstats latest(_value) as ultimo_valor WHERE index=kepler_metrics metric_name=* earliest=-30d by metric_name"
)

print(f"Tipos de métricas: {len(metricas)}")
print("Métricas disponibles:")
for metrica in metricas['metric_name'].tolist():
    print(f"  - {metrica}")
```

**Resultado validado:**
```
Tipos de métricas: 16
Métricas disponibles:
  - flow_rate.SENSOR_003
  - power_consumption.SENSOR_002
  - vibration.SENSOR_007
  - temperature.SENSOR_001
  [... y 12 más]
```

#### **Control de Tiempo Avanzado**

```python
import pandas as pd

# Comparar diferentes rangos temporales
rangos = ["-1h", "-24h", "-7d", "-30d"]
resultados = {}

for rango in rangos:
    datos = kp.data.from_splunk(
        spl="search index=kepler_lab",
        earliest=rango,
        latest="now"
    )
    resultados[rango] = len(datos)
    
print("Datos por rango temporal:")
for rango, cantidad in resultados.items():
    print(f"  {rango}: {cantidad} registros")
```

**Resultado validado:**
```
Datos por rango temporal:
  -1h: 0 registros
  -24h: 90 registros  
  -7d: 2890 registros
  -30d: 2890 registros
```

### **SPL Personalizado Avanzado**

#### **Análisis de Series Temporales**

```python
# Series temporales de métricas por hora
series_temporales = kp.data.from_splunk(
    spl="""
    | mstats avg(_value) as promedio 
    WHERE index=kepler_metrics metric_name=* earliest=-7d 
    span=1h by metric_name
    """
)

# Procesar para análisis
series_temporales['timestamp'] = pd.to_datetime(series_temporales['_time'])
series_temporales['promedio'] = pd.to_numeric(series_temporales['promedio'])

# Estadísticas por métrica
stats = series_temporales.groupby('metric_name')['promedio'].agg([
    'mean', 'std', 'min', 'max', 'count'
])

print("Estadísticas por métrica:")
print(stats.head())
```

#### **Análisis de Eventos Complejos**

```python
# Análisis multi-sensor con filtros
analisis_sensores = kp.data.from_splunk(
    spl="""
    search index=kepler_lab 
    | eval hour=strftime(_time, "%H")
    | stats avg(value) as promedio, count as eventos by sensor_type, area, hour
    | where eventos > 5
    """,
    earliest="-7d"
)

print("Análisis por sensor, área y hora:")
print(analisis_sensores.head(10))
```

### **Manejo de Errores Inteligente**

```python
# El SDK captura y muestra errores de Splunk claramente
try:
    datos = kp.data.from_splunk(
        spl="| mstats avg(_value) WHERE index=kepler_metrics"  # Query incompleta
    )
except Exception as e:
    print(f"Error capturado: {e}")
    # El framework muestra error específico de Splunk + sugerencia
```

**Salida del manejo de errores:**
```
❌ Splunk Error: You must include at least one metric_name filter
🔍 Query: | mstats avg(_value) WHERE index=kepler_metrics
💡 Tip: Check the SPL syntax according to Splunk documentation
```

---

## 📊 Casos de Uso Completos

### **Caso 1: Análisis Exploratorio de Datos (Científico de Datos)**

```python
import kepler as kp
import pandas as pd
import matplotlib.pyplot as plt

# 1. Explorar qué datos hay disponibles
print("=== EXPLORACIÓN INICIAL ===")

# Ver métricas disponibles  
metricas_disponibles = kp.data.from_splunk(
    spl="| mcatalog values(metric_name) WHERE index=kepler_metrics"
)
print(f"Métricas disponibles: {len(metricas_disponibles)}")

# Ver eventos recientes
eventos_recientes = kp.data.from_splunk(
    spl="search index=kepler_lab | head 100"
)
print(f"Tipos de sensores: {eventos_recientes['sensor_type'].unique()}")

# 2. Análisis temporal
print("\n=== ANÁLISIS TEMPORAL ===")

# Comparar volúmenes por período
periodos = {
    "Última hora": "-1h",
    "Últimas 24h": "-24h", 
    "Últimos 7 días": "-7d"
}

for nombre, rango in periodos.items():
    datos = kp.data.from_splunk(
        spl="search index=kepler_lab",
        earliest=rango
    )
    print(f"{nombre}: {len(datos)} registros")

# 3. Análisis por tipo de sensor
print("\n=== ANÁLISIS POR SENSOR ===")

temperatura_data = kp.data.from_splunk(
    spl="search index=kepler_lab sensor_type=temperature",
    earliest="-7d"
)

print(f"Datos de temperatura: {len(temperatura_data)} registros")
print(f"Rango temporal: {temperatura_data['_time'].min()} a {temperatura_data['_time'].max()}")
print(f"Estadísticas básicas:")
print(temperatura_data['value'].describe())
```

### **Caso 2: Pipeline de Datos Automatizado (DevOps/Engineer)**

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

### **Caso 3: Monitoreo en Tiempo Real (Jupyter Notebook)**

```python
# Notebook: real_time_monitoring.ipynb

import kepler as kp
import time
from datetime import datetime

print("=== MONITOR EN TIEMPO REAL ===")

def monitor_loop():
    """Monitor continuo de métricas críticas"""
    
    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Checking metrics...")
        
        # Obtener métricas de los últimos 15 minutos
        metricas_recientes = kp.data.from_splunk(
            spl="| mstats latest(_value) as valor WHERE index=kepler_metrics metric_name=power_consumption.* earliest=-15m by metric_name"
        )
        
        if len(metricas_recientes) > 0:
            # Detectar anomalías simples
            for _, row in metricas_recientes.iterrows():
                metrica = row['metric_name']
                valor = float(row['valor'])
                
                # Umbral simple de ejemplo
                if valor > 1000:  # Consumo alto
                    print(f"⚠️  ALERTA: {metrica} = {valor:.2f} (Alto consumo)")
                elif valor < 100:  # Consumo muy bajo  
                    print(f"🔍 INFO: {metrica} = {valor:.2f} (Consumo bajo)")
                else:
                    print(f"✅ OK: {metrica} = {valor:.2f}")
        else:
            print("❌ No se encontraron métricas recientes")
            
        # Esperar 60 segundos antes del próximo check
        time.sleep(60)

# Ejecutar monitor (detener con Ctrl+C)
try:
    monitor_loop()
except KeyboardInterrupt:
    print("\nMonitor detenido por el usuario")
```

---

## 🔄 Evolución y Roadmap

### **✅ SPRINT 1-8: FUNDACIÓN (COMPLETADO)**

#### **CLI Implementado:**
- ✅ `kepler validate` - Validación completa en 5 pasos
- ✅ `kepler extract` - Extracción con SPL personalizado
- ✅ `kepler config` - Gestión de configuración

#### **SDK Implementado:**
- ✅ `import kepler as kp` - Import directo
- ✅ `kp.data.from_splunk()` - Extracción flexible
- ✅ Parámetros `earliest`/`latest` para control temporal
- ✅ Manejo inteligente de errores de Splunk

#### **Capacidades Validadas:**
- ✅ **2,890 eventos** extraídos exitosamente
- ✅ **16 tipos de métricas** funcionando
- ✅ **Control temporal:** 32x diferencia entre 24h vs 7d
- ✅ **Notebooks Jupyter** integración completa

### **🚧 SPRINT 9-10: ENTRENAMIENTO ML (EN DESARROLLO)**

#### **CLI Nuevo:**
```bash
# Comandos de ML que se van a implementar
kepler train data.csv --target temperature --algorithm random_forest
kepler train data.csv --target anomaly --algorithm xgboost --params config.json
kepler model list  # Listar modelos entrenados
kepler model info model_20241215_rf.pkl  # Info del modelo
```

#### **SDK Nuevo:**
```python
# API de ML que se va a implementar
model = kp.train.sklearn(
    data=eventos,
    target='temperature', 
    algorithm='RandomForest',
    test_size=0.3
)

# Predicciones locales
predictions = model.predict(test_data)
print(f"Accuracy: {model.metrics['accuracy']}")
```

### **🚧 SPRINT 11-12: DEPLOYMENT GCP (PLANEADO)**

#### **CLI Deployment:**
```bash
# Deploy a Cloud Run
kepler deploy model.pkl --name temp-predictor --env production

# Gestión de deployments  
kepler deploy list
kepler deploy logs temp-predictor --tail
kepler deploy scale temp-predictor --instances 3
```

#### **SDK Deployment:**
```python
# Deploy programático
endpoint = kp.deploy.cloud_run(
    model=trained_model,
    name="anomaly-detector",
    environment="production"
)

# Predicciones remotas
result = endpoint.predict({"sensor_data": [25.5, 2.1, 0.8]})
```

### **🚧 SPRINT 13-16: PRODUCCIÓN COMPLETA (FUTURO)**

#### **Funcionalidades Avanzadas:**
- 🔄 **Escritura automática** de predicciones a Splunk
- 📊 **Dashboards automáticos** en Splunk para monitoreo
- 🔧 **Pipeline completos** de CI/CD
- 📦 **Distribución PyPI** (`pip install kepler-framework`)

---

## 📚 Recursos Adicionales

### **Notebooks de Ejemplo (Validados):**
- `test-lab/notebooks/metrics_analysis_clean.ipynb` - Análisis de métricas paso a paso
- `test-lab/notebooks/events_analysis.ipynb` - Análisis de eventos completo

### **Archivos de Configuración:**
- `~/.kepler/config.yml` - Configuración global segura
- `proyecto/kepler.yml` - Configuración específica del proyecto

### **Documentación Técnica:**
- `README.md` - Guía de inicio rápido
- `VALIDATION_STATUS.md` - Estado técnico detallado
- Este archivo: `SDK_CLI_GUIDE.md` - Guía completa de uso

---

> **💡 Tip Final:** Comienza siempre con `kepler validate` para asegurar que todo esté configurado correctamente. Luego usa `kepler extract` para explorar tus datos antes de entrenar modelos.

> **🎯 Estado Actual:** Las funciones de extracción y análisis están completamente validadas y listas para uso en producción. Las funciones de ML están en desarrollo activo.