# Kepler Framework - Guía AutoML

> **Última actualización:** 7 de Septiembre de 2025  
> **Estado:** Completamente Implementado (Tasks 1.11-1.15)  
> **Audiencia:** Científicos de Datos, Analistas, Ingenieros de IA

## 🎯 Sistema AutoML Completo

Kepler proporciona capacidades AutoML avanzadas para automatizar la selección de algoritmos, optimización de hiperparámetros, feature engineering y evaluación de modelos.

### **Filosofía: "Experimentación Automática + Control Manual"**

El sistema AutoML de Kepler está diseñado para:
- **Acelerar el desarrollo** de modelos mediante automatización inteligente
- **Mantener transparencia** en todas las decisiones automáticas
- **Permitir control manual** cuando sea necesario
- **Soportar constraints industriales** para entornos de producción

---

## 🚀 APIs Principales

### **1. Selección Automática de Algoritmos**

```python
import kepler as kp

# Selección automática basada en características de datos
best_algo = kp.automl.select_algorithm(data, target="failure")
print(f"Algoritmo recomendado: {best_algo}")

# Obtener top-3 recomendaciones con scores
recommendations = kp.automl.recommend_algorithms(data, target="failure", top_k=3)
for rec in recommendations:
    print(f"{rec['algorithm']}: {rec['score']:.3f} - {rec['reason']}")
```

### **2. Entrenamiento Automático**

```python
# AutoML básico - selecciona y entrena automáticamente
model = kp.automl.auto_train(data, target="failure")
print(f"Modelo entrenado: {model.algorithm}")
print(f"Performance: {model.performance}")

# AutoML con constraints
model = kp.automl.auto_train(
    data, 
    target="failure",
    constraints={
        "interpretability": "high",
        "max_training_time": "30m",
        "max_model_size": "50MB"
    }
)
```

### **3. Optimización de Hiperparámetros**

```python
# Optimización automática con Optuna
optimized_params = kp.automl.optimize_hyperparameters(
    data, 
    target="failure", 
    algorithm="xgboost",
    n_trials=100,
    timeout=3600  # 1 hora
)

print(f"Mejores parámetros: {optimized_params['best_params']}")
print(f"Mejor score: {optimized_params['best_score']:.4f}")
```

### **4. Feature Engineering Automático**

```python
# Feature engineering automático
engineered_data = kp.automl.engineer_features(
    data,
    target="failure",
    polynomial_features=True,
    interaction_features=True,
    feature_selection=True
)

print(f"Features originales: {data.shape[1]}")
print(f"Features después de engineering: {engineered_data.shape[1]}")
```

### **5. Suite de Experimentos Paralelos**

```python
# Ejecutar múltiples experimentos en paralelo
experiment_results = kp.automl.run_experiment_suite(
    data,
    target="failure",
    algorithms=["xgboost", "lightgbm", "pytorch", "transformers"],
    parallel_jobs=4,
    optimization_budget="2h"
)

# Ver leaderboard automático
leaderboard = kp.automl.get_experiment_leaderboard(experiment_results)
print(leaderboard)
```

### **6. Pipeline AutoML Completo**

```python
# Pipeline end-to-end automático
pipeline_result = kp.automl.automl_pipeline(
    data,
    target="equipment_failure",
    optimization_time="1h",
    interpretability_required=True,
    max_inference_latency_ms=200,
    max_model_size_mb=50
)

# Resultados completos
best_model = pipeline_result['best_model']
deployment_package = pipeline_result['deployment_ready']
print(f"Mejor modelo: {pipeline_result['best_algorithm']}")
print(f"Score: {pipeline_result['best_score']:.3f}")
```

### **7. AutoML Industrial**

```python
# AutoML específico para casos industriales
industrial_result = kp.automl.industrial_automl(
    sensor_data,
    target="equipment_failure",
    use_case="predictive_maintenance",  # Constraints predefinidos
    optimization_budget="30m",
    production_environment="edge"  # Optimización para edge
)

if industrial_result['deployment_ready']:
    print(f"✅ Listo para producción: {industrial_result['best_algorithm']}")
    print(f"Latencia esperada: {industrial_result['expected_latency_ms']}ms")
else:
    print(f"⚠️ Requiere revisión: {industrial_result['issues']}")
```

---

## 🎯 Casos de Uso Prácticos

### **Caso 1: Científico Nuevo en el Proyecto**

```python
# Ana quiere entrenar un modelo pero no sabe qué algoritmo usar
import kepler as kp

# Carga datos
data = kp.data.from_splunk("search index=sensors", time_range="-30d")

# AutoML hace todo automáticamente
model = kp.automl.auto_train(data, target="equipment_failure")

# Resultado: Mejor modelo seleccionado y entrenado automáticamente
print(f"AutoML seleccionó: {model.algorithm}")
print(f"Accuracy: {model.performance['accuracy']:.3f}")
```

### **Caso 2: Optimización para Producción**

```python
# Modelo para entorno industrial con constraints estrictos
production_model = kp.automl.industrial_automl(
    data,
    target="critical_failure",
    use_case="predictive_maintenance",
    production_environment="edge",  # Dispositivo edge
    optimization_budget="2h"
)

# Validar que cumple constraints industriales
if production_model['deployment_ready']:
    print("✅ Cumple todos los constraints industriales")
    print(f"Latencia: {production_model['expected_latency_ms']}ms")
    print(f"Tamaño: {production_model['model_size_mb']}MB")
    print(f"Interpretabilidad: {production_model['interpretability_score']}")
```

### **Caso 3: Comparación de Múltiples Algoritmos**

```python
# Comparar rendimiento de diferentes enfoques
experiment = kp.automl.run_experiment_suite(
    data,
    target="failure",
    algorithms=["sklearn", "xgboost", "pytorch", "transformers"],
    parallel_jobs=2,
    optimization_budget="3h"
)

# Ver resultados comparativos
leaderboard = kp.automl.get_experiment_leaderboard(experiment)
print(leaderboard)

# Seleccionar el mejor para producción
best_model = experiment['models'][experiment['best_algorithm']]
```

---

## 📊 Versionado y Reproducibilidad

### **Versionado Automático de Experimentos AutoML**

```python
# Los experimentos AutoML se versionan automáticamente
with kp.versioning.track_experiment("automl-optimization") as exp:
    # AutoML registra automáticamente:
    # - Datos utilizados y su versión
    # - Algoritmos probados y parámetros
    # - Métricas de cada experimento
    # - Mejor modelo seleccionado
    
    result = kp.automl.automl_pipeline(data, target="failure")
    
    # Crear release del mejor modelo
    if result['deployment_ready']:
        release = kp.versioning.create_release(
            f"automl-best-{result['best_algorithm']}-v1.0"
        )
```

### **Reproducibilidad de Experimentos AutoML**

```python
# Reproducir experimento AutoML exacto
reproduction = kp.reproduce.from_version("automl-xgboost-v1.0")

# Kepler reproduce automáticamente:
# ✅ Mismos datos (versionados con DVC)
# ✅ Mismo algoritmo y parámetros
# ✅ Mismas features engineered
# ✅ Mismo environment de librerías
# ✅ Misma semilla aleatoria

assert reproduction.success
assert reproduction.metadata['model_performance'] == original_performance
```

---

## 🔄 Evolución del API
