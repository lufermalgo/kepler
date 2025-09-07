# Kepler Framework - Guía SDK (Python API)

> **Versión del Framework:** 0.1.0  
> **Última actualización:** 6 de Septiembre de 2025  
> **Estado:** API Core Validada  
> **Audiencia:** Científicos de Datos, Analistas, Desarrolladores Python

## 📋 Tabla de Contenidos

1. [Introducción al SDK](#introducción-al-sdk)
2. [Instalación y Setup](#instalación-y-setup)
3. [API de Extracción de Datos](#api-de-extracción-de-datos)
4. [Jupyter Notebooks](#jupyter-notebooks)
5. [Casos de Uso Avanzados](#casos-de-uso-avanzados)
6. [Evolución del API](#evolución-del-api)

---

## 🎯 Introducción al SDK

El **SDK de Kepler** está diseñado para:

### **🐍 Análisis Interactivo**
- Integración nativa con Jupyter notebooks
- API simple y familiar (similar a pandas/sklearn)
- Import directo sin configuración manual

### **📊 Ciencia de Datos**
- Extracción flexible de datos desde Splunk
- Control temporal preciso para análisis históricos
- Integración con ecosystem Python (pandas, numpy, matplotlib)

### **🔬 Experimentación**
- Prototipado rápido de modelos
- Análisis exploratorio de datos industriales
- Validación de hipótesis con datos reales

---

## ⚙️ Instalación y Setup

### Instalación Básica

```python
# Instalar Kepler (desde terminal)
# git clone https://github.com/lufermalgo/kepler.git /tmp/kepler-install  
# cd /tmp/kepler-install && pip install .
# rm -rf /tmp/kepler-install

# Import súper simple - sin configuración manual
import kepler as kp

print(f"Kepler Framework version: {kp.__version__}")
```

### Configuración Automática

El SDK usa automáticamente la configuración de `~/.kepler/config.yml`:

```python
# No necesitas configurar nada manualmente
# El SDK carga automáticamente:
# - Tokens de Splunk
# - Configuración GCP  
# - Preferencias de conexión
```

---

## 📊 API de Extracción de Datos

### **Función Principal: `kp.data.from_splunk()`**

#### **Extracción Básica de Eventos**

```python
import kepler as kp

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

### **Control de Tiempo Avanzado**

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

## 📓 Jupyter Notebooks

### **Setup en Notebook**

```python
# Celda 1: Import y configuración
import kepler as kp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuración para notebooks (opcional)
import os
os.environ['KEPLER_NOTEBOOK_MODE'] = 'true'  # Salida limpia

print(f"✅ Kepler {kp.__version__} cargado correctamente")
```

### **Análisis Exploratorio Básico**

```python
# Celda 2: Exploración inicial
print("=== EXPLORACIÓN INICIAL ===")

# Ver qué datos hay disponibles
eventos_recientes = kp.data.from_splunk(
    spl="search index=kepler_lab | head 100"
)

print(f"Tipos de sensores: {eventos_recientes['sensor_type'].unique()}")
print(f"Áreas disponibles: {eventos_recientes['area'].unique()}")
print(f"Rango temporal: {eventos_recientes['_time'].min()} a {eventos_recientes['_time'].max()}")
```

### **Visualización de Datos**

```python
# Celda 3: Visualización
temperatura_data = kp.data.from_splunk(
    spl="search index=kepler_lab sensor_type=temperature",
    earliest="-7d"
)

# Convertir timestamp
temperatura_data['timestamp'] = pd.to_datetime(temperatura_data['_time'])
temperatura_data['value'] = pd.to_numeric(temperatura_data['value'])

# Gráfico de series temporales
plt.figure(figsize=(12, 6))
for area in temperatura_data['area'].unique():
    data_area = temperatura_data[temperatura_data['area'] == area]
    plt.plot(data_area['timestamp'], data_area['value'], 
             label=f'Área {area}', alpha=0.7)

plt.title('Temperatura por Área - Últimos 7 días')
plt.xlabel('Tiempo')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 🔬 Casos de Uso Avanzados

### **Caso 1: Análisis de Correlación**

```python
# Obtener datos de diferentes tipos de sensores
temperatura = kp.data.from_splunk(
    spl="search index=kepler_lab sensor_type=temperature",
    earliest="-30d"
)

presion = kp.data.from_splunk(
    spl="search index=kepler_lab sensor_type=pressure", 
    earliest="-30d"
)

# Análisis de correlación por área
areas = temperatura['area'].unique()

for area in areas:
    temp_area = temperatura[temperatura['area'] == area]['value'].astype(float)
    pres_area = presion[presion['area'] == area]['value'].astype(float)
    
    if len(temp_area) > 10 and len(pres_area) > 10:
        correlation = np.corrcoef(temp_area[:min(len(temp_area), len(pres_area))], 
                                pres_area[:min(len(temp_area), len(pres_area))])[0,1]
        print(f"Correlación Temperatura-Presión en {area}: {correlation:.3f}")
```

### **Caso 2: Detección de Anomalías Simple**

```python
# Obtener métricas de consumo de energía
energia = kp.data.from_splunk(
    spl="| mstats avg(_value) as consumo WHERE index=kepler_metrics metric_name=power_consumption.* earliest=-7d by metric_name",
)

# Detectar valores anómalos (> 3 desviaciones estándar)
for _, row in energia.iterrows():
    sensor = row['metric_name']
    valor = float(row['consumo'])
    
    # Obtener historial para estadísticas
    historial = kp.data.from_splunk(
        spl=f"| mstats avg(_value) as valor WHERE index=kepler_metrics metric_name={sensor} earliest=-30d span=1h by _time"
    )
    
    if len(historial) > 10:
        valores = pd.to_numeric(historial['valor'])
        media = valores.mean()
        std = valores.std()
        
        if abs(valor - media) > 3 * std:
            print(f"⚠️ ANOMALÍA DETECTADA: {sensor}")
            print(f"   Valor actual: {valor:.2f}")
            print(f"   Media histórica: {media:.2f} ± {std:.2f}")
```

### **Caso 3: Análisis de Tendencias**

```python
from scipy import stats

# Analizar tendencia de temperatura en el tiempo
temperatura_trend = kp.data.from_splunk(
    spl="""
    search index=kepler_lab sensor_type=temperature 
    | bucket _time span=1d 
    | stats avg(value) as temp_promedio by _time
    """,
    earliest="-30d"
)

# Convertir datos
temperatura_trend['timestamp'] = pd.to_datetime(temperatura_trend['_time'])
temperatura_trend['temp_promedio'] = pd.to_numeric(temperatura_trend['temp_promedio'])

# Calcular tendencia
x = np.arange(len(temperatura_trend))
y = temperatura_trend['temp_promedio'].values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Tendencia de temperatura:")
print(f"  Pendiente: {slope:.4f} °C/día")
print(f"  Correlación: {r_value:.3f}")
print(f"  P-value: {p_value:.3f}")

if p_value < 0.05:
    if slope > 0:
        print("  📈 Tendencia al ALZA significativa")
    else:
        print("  📉 Tendencia a la BAJA significativa")
else:
    print("  ➡️ Sin tendencia significativa")
```

---

## 🚧 API en Desarrollo

### **Sprint Actual - ML Training**

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

### **Sprint 11-12 - Deployment**

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

---

## 📚 Notebooks de Ejemplo

### **Notebooks Validados Disponibles:**
- **`test-lab/notebooks/metrics_analysis_clean.ipynb`** - Análisis de métricas paso a paso
- **`test-lab/notebooks/events_analysis.ipynb`** - Análisis de eventos completo

### **Estructura Típica de Notebook:**

```python
# Celda 1: Setup
import kepler as kp
import pandas as pd
import matplotlib.pyplot as plt

# Celda 2: Exploración
datos = kp.data.from_splunk(spl="search index=kepler_lab", earliest="-7d")
print(f"Datos extraídos: {len(datos)}")

# Celda 3: Análisis
# Tu análisis específico aquí...

# Celda 4: Visualización  
# Gráficos y plots...

# Celda 5: Conclusiones
# Insights y próximos pasos...
```

---

## 🤖 API de Entrenamiento Unificada

### **Función Principal: `kp.train_unified.train()`**

```python
import kepler as kp

# API unificada para cualquier framework AI
model = kp.train_unified.train(data, target="failure", algorithm="auto")

# Traditional ML
model = kp.train_unified.train(data, target="failure", algorithm="xgboost")
model = kp.train_unified.train(data, target="failure", algorithm="random_forest")

# Deep Learning  
model = kp.train_unified.train(data, target="failure", algorithm="pytorch", epochs=100)

# Generative AI
model = kp.train_unified.train(text_data, target="sentiment", algorithm="transformers", 
                              text_column="review_text")
```

### **AutoML Inteligente: `kp.automl.*`**

```python
# Selección automática de algoritmo
best_algo = kp.automl.select_algorithm(data, target="failure")
print(f"Mejor algoritmo: {best_algo}")

# Entrenamiento automático completo
model = kp.automl.auto_train(data, target="failure")

# AutoML con constraints industriales
industrial_result = kp.automl.industrial_automl(
    data, 
    target="equipment_failure",
    use_case="predictive_maintenance",
    optimization_budget="1h"
)
```

### **Sistema de Versionado MLOps: `kp.versioning.*`**

```python
# Crear versión unificada (Git + DVC + MLflow)
version = kp.versioning.create_unified_version(
    "production-v1.0",
    data_paths=["data/sensors.csv"],
    experiment_name="predictive-maintenance"
)

# Reproducir cualquier versión
result = kp.reproduce.from_version("production-v1.0")

# Gestión de releases
release = kp.versioning.create_release("stable-v1.0", status="production")
```

### **Gestión Ilimitada de Librerías: `kp.libs.*`**

```python
# Instalar cualquier librería Python
kp.libs.install("transformers>=4.30.0")
kp.libs.install("git+https://github.com/research/experimental-ai.git")

# Crear templates de AI
kp.libs.template("generative_ai")  # Instala transformers, langchain, etc.
kp.libs.template("deep_learning")  # Instala pytorch, tensorflow, etc.

# Validar entorno
status = kp.libs.validate()
```

---

## 🔄 Evolución del API

### **✅ Funcionalidades Actuales (0.1.0)**
- ✅ `kp.data.from_splunk()` - Extracción completa de datos
- ✅ `kp.train_unified.train()` - API unificada para cualquier framework AI
- ✅ `kp.automl.*` - Sistema AutoML completo (selección automática, optimización)
- ✅ `kp.versioning.*` - Sistema MLOps completo (Git + DVC + MLflow)
- ✅ `kp.reproduce.from_version()` - Reproducibilidad completa
- ✅ `kp.libs.*` - Soporte ilimitado de librerías Python
- ✅ Control temporal con `earliest`/`latest`
- ✅ Manejo inteligente de errores con códigos estándar
- ✅ Integración Jupyter optimizada

### **🚧 Próximas Versiones**
- 🔄 `kp.deploy.*` - APIs de deployment automático
- 🔄 `kp.monitor.*` - APIs de monitoreo híbrido
- 🔄 `kp.validate.*` - Validación completa de ecosistemas
- 🔄 `kp.docs.*` - Generación automática de documentación

---

## 📋 Recursos Adicionales

### **Documentación Relacionada**
- **[CLI Guide](./CLI_GUIDE.md)** - Comandos de línea para automatización
- **[Estado de Validación](./VALIDATION_STATUS.md)** - Funcionalidades probadas

### **Ejemplos Prácticos**
- Notebooks en `test-lab/notebooks/`
- Scripts de ejemplo en `test-lab/scripts/`

### **Soporte y Comunidad**
- Issues en GitHub para reportar problemas
- Documentación actualizada continuamente

---

> **💡 Tip para Científicos:** El SDK está optimizado para análisis exploratorio. Usa `earliest` y `latest` para controlar exactamente qué datos necesitas para tu análisis.

> **🎯 Estado Actual:** La API de extracción está completamente validada con datos reales (2,890 eventos + 16 métricas) y lista para uso en análisis de producción.