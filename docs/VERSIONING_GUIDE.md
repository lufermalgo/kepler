# Kepler Framework - Guía de Versionado MLOps

> **Última actualización:** 7 de Septiembre de 2025  
> **Estado:** Completamente Implementado (M2 - Tasks 5.1-5.7)  
> **Audiencia:** Científicos de Datos, Ingenieros MLOps, DevOps

## 🎯 Sistema de Versionado Completo

Kepler implementa un sistema de versionado MLOps completo que integra **Git + DVC + MLflow** para garantizar reproducibilidad total desde datos hasta deployment.

### **Filosofía: "Trazabilidad Total - Datos + Código + Modelos + Features"**

El sistema de versionado de Kepler rastrea:
- ✅ **Datos**: Versiones de datasets con DVC
- ✅ **Código**: Commits de Git con trazabilidad
- ✅ **Modelos**: Registry de MLflow con métricas
- ✅ **Features**: Pipelines de transformación versionados
- ✅ **Experimentos**: Tracking completo de experimentos
- ✅ **Releases**: Versionado multi-componente para producción

---

## 🚀 APIs Principales

### **1. Versionado de Datos**

```python
import kepler as kp

# Versionar dataset automáticamente
dataset_version = kp.versioning.version_data(
    "data/sensors.csv",
    version_name="sensors-v1.0",
    metadata={"source": "splunk", "rows": 2890, "features": 15}
)

# Listar versiones disponibles
versions = kp.versioning.list_versions()
for v in versions:
    print(f"{v.version_id}: {v.file_path} ({v.rows} rows)")
```

### **2. Versionado de Feature Engineering**

```python
# Crear pipeline de features versionado
pipeline = kp.versioning.version_feature_pipeline(
    pipeline_name="sensor-preprocessing",
    steps=[
        {"type": "StandardScaler"},
        {"type": "PolynomialFeatures", "degree": 2},
        {"type": "SelectKBest", "k": 10}
    ],
    version="v1.2.0"
)

# Aplicar pipeline versionado
transformed_data = kp.versioning.apply_versioned_pipeline(
    data, 
    pipeline_name="sensor-preprocessing",
    version="v1.2.0"
)
```

### **3. Tracking de Experimentos con MLflow**

```python
# Iniciar experimento con tracking automático
experiment_id = kp.versioning.start_experiment("predictive-maintenance")

# Registrar parámetros y métricas automáticamente
kp.versioning.log_parameters({"algorithm": "xgboost", "n_estimators": 100})
kp.versioning.log_metrics({"accuracy": 0.94, "f1_score": 0.93})

# Finalizar experimento
result = kp.versioning.end_experiment()
print(f"Experiment ID: {result['run_id']}")
```

### **4. Versionado Unificado (Git + DVC + MLflow)**

```python
# Crear versión unificada que integra todo
unified_version = kp.versioning.create_unified_version(
    version_name="production-v1.0",
    data_paths=["data/sensors.csv", "data/maintenance.csv"],
    experiment_name="predictive-maintenance",
    metadata={
        "model_type": "xgboost",
        "performance": {"accuracy": 0.94},
        "deployment_target": "cloud_run"
    }
)

print(f"Versión unificada creada: {unified_version.version_id}")
print(f"Git commit: {unified_version.git_commit}")
print(f"DVC version: {unified_version.dvc_data_version}")
print(f"MLflow run: {unified_version.mlflow_run_id}")
```

### **5. Sistema de Reproducción Completa**

```python
# Reproducir cualquier versión exacta
reproduction_result = kp.reproduce.from_version("production-v1.0")

if reproduction_result.success:
    print("✅ Reproducción exitosa")
    print(f"Pasos completados: {reproduction_result.steps_completed}")
    print(f"Artifacts creados: {reproduction_result.artifacts_created}")
else:
    print(f"❌ Error en reproducción: {reproduction_result.error_message}")

# Reproducir tipos específicos
data_reproduction = kp.reproduce.from_version("sensors@v1.0", reproduction_type="data")
pipeline_reproduction = kp.reproduce.from_version("preprocessing@v1.2", reproduction_type="pipeline")
experiment_reproduction = kp.reproduce.from_version("experiment-123", reproduction_type="experiment")
```

### **6. Gestión de Releases**

```python
# Crear release para producción
release = kp.versioning.create_release(
    release_name="stable-v1.0",
    version="1.0.0",
    components={
        "data_version": "sensors-v1.0",
        "model_version": "xgboost-v1.3",
        "pipeline_version": "preprocessing-v1.2"
    },
    status="production"
)

# Promover release
kp.versioning.promote_release("stable-v1.0", target_environment="production")

# Reproducir release completo
release_reproduction = kp.versioning.reproduce_release("stable-v1.0")
```

### **7. Trazabilidad End-to-End**

```python
# Crear trazabilidad completa de datos a deployment
data_lineage = kp.versioning.create_data_lineage(
    node_id="sensors-raw",
    data_path="data/sensors.csv",
    metadata={"source": "splunk", "extraction_time": "2025-09-07T08:00:00Z"}
)

pipeline_lineage = kp.versioning.create_pipeline_lineage(
    node_id="preprocessing-v1.2",
    pipeline_name="sensor-preprocessing",
    inputs=["sensors-raw"],
    metadata={"steps": ["scaler", "polynomial", "selection"]}
)

model_lineage = kp.versioning.create_model_lineage(
    node_id="xgboost-v1.3",
    model_name="predictive-maintenance",
    inputs=["preprocessing-v1.2"],
    metadata={"algorithm": "xgboost", "accuracy": 0.94}
)

# Obtener trazabilidad completa
complete_lineage = kp.versioning.get_complete_lineage("xgboost-v1.3")
print(f"Trazabilidad completa desde datos hasta modelo:")
for step in complete_lineage:
    print(f"  {step['node_id']} → {step['outputs']}")
```

---

## 🏭 AutoML Industrial

### **Constraints Predefinidos por Caso de Uso**

```python
# Predictive Maintenance
maintenance_result = kp.automl.industrial_automl(
    data,
    target="equipment_failure", 
    use_case="predictive_maintenance",  # Constraints automáticos:
    # - Interpretabilidad alta (sklearn, xgboost)
    # - Latencia <100ms
    # - Tamaño <50MB
    # - Robustez a outliers
)

# Quality Control
quality_result = kp.automl.industrial_automl(
    data,
    target="defect_detected",
    use_case="quality_control",  # Constraints automáticos:
    # - Precisión muy alta (recall prioritario)
    # - Modelos explicables
    # - Tiempo real <50ms
)

# Anomaly Detection
anomaly_result = kp.automl.industrial_automl(
    data,
    target="anomaly",
    use_case="anomaly_detection",  # Constraints automáticos:
    # - Detección de outliers
    # - Baja tasa de falsos positivos
    # - Adaptación a concept drift
)
```

### **Optimización por Entorno**

```python
# Edge Computing
edge_model = kp.automl.industrial_automl(
    data,
    target="failure",
    production_environment="edge",  # Optimizaciones:
    # - Modelos ligeros (<10MB)
    # - CPU únicamente
    # - Baja latencia (<50ms)
    # - Offline capability
)

# Cloud Computing
cloud_model = kp.automl.industrial_automl(
    data,
    target="failure", 
    production_environment="cloud",  # Optimizaciones:
    # - Modelos más complejos permitidos
    # - GPU acceleration disponible
    # - Latencia relajada (<200ms)
    # - Escalabilidad automática
)
```

---

## 🔄 Integración con Workflow Completo

### **Pipeline Típico: Datos → AutoML → Versionado → Deployment**

```python
# 1. Extraer y versionar datos
data = kp.data.from_splunk("search index=sensors", time_range="-7d")
data_version = kp.versioning.version_data("data/sensors-latest.csv", "sensors-v2.0")

# 2. AutoML con versionado automático
with kp.versioning.track_experiment("automl-v2.0") as exp:
    automl_result = kp.automl.automl_pipeline(
        data,
        target="failure",
        optimization_time="1h",
        interpretability_required=True
    )
    
    # 3. Crear versión unificada del mejor modelo
    unified_version = kp.versioning.create_unified_version(
        f"automl-best-{automl_result['best_algorithm']}-v2.0",
        data_paths=["data/sensors-latest.csv"],
        experiment_name="automl-v2.0"
    )

# 4. Crear release para producción
if automl_result['deployment_ready']:
    release = kp.versioning.create_release(
        f"production-{automl_result['best_algorithm']}-v2.0",
        version="2.0.0",
        status="ready_for_deployment"
    )
    
    print(f"✅ Release creado: {release.release_id}")
    print(f"Listo para: kepler deploy {release.release_id}")
```

---

## 📋 Comandos CLI

### **Versionado desde CLI**

```bash
# Crear versión unificada
kepler versioning create-unified "production-v1.0" \
  --data-paths "data/sensors.csv" \
  --experiment "predictive-maintenance"

# Listar versiones
kepler versioning list-unified

# Reproducir versión
kepler reproduce from-version "production-v1.0"

# Crear release
kepler versioning create-release "stable-v1.0" \
  --version "1.0.0" \
  --status "production"
```

### **AutoML desde CLI**

```bash
# AutoML automático
kepler automl run data.csv --target failure \
  --optimization-time "1h" \
  --constraints-file industrial.json

# Pipeline AutoML industrial
kepler automl industrial data.csv --target failure \
  --use-case "predictive_maintenance" \
  --environment "edge"
```

---

## 🎯 Mejores Prácticas

### **1. Versionado Sistemático**
- Versiona datos antes de cada experimento mayor
- Usa nombres descriptivos para versiones (`sensors-maintenance-v1.0`)
- Incluye metadata relevante (source, quality, transformations)

### **2. Experimentos Reproducibles**
- Siempre usa `track_experiment()` para experimentos importantes
- Documenta decisiones de AutoML en metadata
- Crea releases solo de modelos validados

### **3. Gestión de Releases**
- Usa semantic versioning (v1.0.0, v1.1.0, v2.0.0)
- Incluye notas de release con cambios importantes
- Valida reproducibilidad antes de promover a producción

---

## 📊 Métricas y Monitoreo

El sistema de versionado integra automáticamente con el monitoreo para rastrear:
- **Data drift**: Cambios en distribución de datos entre versiones
- **Model performance**: Degradación de performance en producción
- **Pipeline stability**: Estabilidad de feature engineering
- **Deployment success**: Tasa de éxito de deployments por versión

---

> **💡 Tip:** Combina AutoML con versionado para experimentación sistemática. El AutoML genera múltiples candidatos, el versionado te permite comparar y reproducir cualquiera.

> **🎯 Estado Actual:** Sistema completo de versionado MLOps implementado y validado. Listo para uso en proyectos de producción.
