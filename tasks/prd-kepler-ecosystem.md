# Product Requirements Document - Kepler Framework Ecosystem

> **Fecha:** 6 de Septiembre de 2025  
> **Versión:** 1.0 (Documento Fundacional)  
> **Estado:** Draft Inicial  
> **Audiencia:** Científicos de Datos, Ingenieros, Analistas, Stakeholders

## 1. Introducción/Overview

**Kepler** es un framework de ecosistema de **Inteligencia Artificial y Ciencia de Datos** agnóstico a tecnologías, diseñado para eliminar completamente las barreras entre **cualquier fuente de datos** y la experimentación libre con **cualquier librería Python** - desde PyPI oficial hasta repositorios privados, librerías experimentales, y desarrollos custom.

**Versión inicial (v0.1-v1.0):** Splunk como fuente de datos + GCP como plataforma de compute  
**Roadmap futuro:** AWS, Azure, on-premises, edge computing, BigQuery, PostgreSQL, APIs REST, archivos, etc. 

### Contexto del Problema

Las organizaciones enfrentan limitaciones críticas con las herramientas actuales de ciencia de datos:

- **Splunk ML Toolkit**: Modelos básicos prefigurados, capacidades limitadas
- **Splunk Deep Learning Toolkit**: Promete TensorFlow/PyTorch en Jupyter, pero presenta:
  - Limitaciones en experimentación real
  - Problemas con entornos containerizados (Docker/K8s)  
  - Restricciones en librerías disponibles
- **ONNX como "escape valve"**: Empaquetado de modelos externos, pero proceso complejo

### El Problema Fundamental

Los científicos de datos siguen teniendo **trabas para experimentar libremente y desplegar fácilmente**, incluso con las soluciones "avanzadas" de Splunk. Necesitan un framework que les permita:

1. **Experimentación sin restricciones** con cualquier librería Python:
   - PyPI oficial (sklearn, transformers, pytorch)
   - Repositorios GitHub/GitLab (experimentales, forks)
   - Librerías corporativas privadas  
   - Desarrollos custom y locales
   - Cualquier fuente de código Python
2. **Gestión automática de ecosistemas** desarrollo/producción
3. **Despliegue sin fricción** de notebook a producción  
4. **Arquitectura modular multi-cloud** sin vendor lock-in

## 2. Goals

### Objetivos Primarios
1. **Maximizar posibilidades de experimentación** para analistas y científicos de datos con **cualquier tecnología de IA**: Machine Learning, Deep Learning, IA Generativa, NLP, Computer Vision, y cualquier librería Python existente o futura
2. **Simplificar el paso a producción** de semanas a días
3. **Gestionar automáticamente ecosistemas** de desarrollo y producción
4. **Eliminar restricciones de licenciamiento** basado en ingesta Splunk
5. **Proporcionar interoperabilidad completa** con entornos de desarrollo modernos

### Objetivos Secundarios  
6. **Reducir costos operativos** mediante procesamiento en cloud vs Splunk
7. **Democratizar herramientas ML avanzadas** sin dependencia de infraestructura propietaria
8. **Crear comunidad open-source** activa con plugins y adaptadores

## 3. User Stories

### Científico de Datos
- **Como** científico de datos, **quiero** importar cualquier librería Python (PyTorch, TensorFlow, sklearn, etc.) **para que** pueda experimentar sin limitaciones técnicas
- **Como** científico de datos, **quiero** extraer datos de Splunk con queries SPL personalizadas **para que** pueda usar exactamente los datos que necesito
- **Como** científico de datos, **quiero** desplegar mis modelos a producción con un comando **para que** no dependa de DevOps para cada deployment

### Ingeniero de Datos  
- **Como** ingeniero de datos, **quiero** gestionar automáticamente entornos de desarrollo/producción **para que** los científicos puedan trabajar independientemente
- **Como** ingeniero de datos, **quiero** monitorear todos los modelos en producción desde Splunk **para que** tenga visibilidad completa del sistema
- **Como** ingeniero de datos, **quiero** configurar pipelines de datos una vez **para que** se reutilicen automáticamente

### Analista de Negocio
- **Como** analista, **quiero** usar herramientas familiares (Jupyter, pandas) **para que** no tenga curva de aprendizaje adicional
- **Como** analista, **quiero** acceso a datos históricos con rangos de tiempo flexibles **para que** pueda hacer análisis retrospectivos
- **Como** analista, **quiero** escribir resultados automáticamente a Splunk **para que** los dashboards se actualicen solos

### Stakeholder Técnico
- **Como** CTO, **quiero** reducir dependencia de soluciones propietarias **para que** tengamos flexibilidad tecnológica
- **Como** gerente de proyecto, **quiero** métricas claras de adopción y ROI **para que** pueda justificar la inversión
- **Como** arquitecto, **quiero** extensibilidad via plugins **para que** podamos adaptar el framework a necesidades específicas

## 3.1. Soporte Ilimitado de Librerías Python

### Filosofía: "Si está en Python, Kepler lo soporta"

Kepler está diseñado para **no limitar nunca** las opciones tecnológicas del científico de datos. El framework debe soportar:

#### Fuentes de Librerías Soportadas
1. **PyPI Oficial**: `pip install numpy`, `pip install transformers`
2. **Repositorios Git**: `pip install git+https://github.com/user/repo.git`
3. **Repositorios Privados**: Con autenticación SSH/HTTPS
4. **Librerías Locales**: `pip install -e ./mi-libreria-custom`
5. **Archivos Wheel/Tar**: `pip install ./libreria-custom-1.0.whl`
6. **Forks Personalizados**: Modificaciones de librerías existentes
7. **Versiones Experimentales**: Alpha, beta, release candidates

#### Ejemplos de Ecosistemas Soportados

```python
# Machine Learning Tradicional
import sklearn, xgboost, lightgbm, catboost

# Deep Learning
import torch, tensorflow, keras, jax, lightning

# IA Generativa
import transformers, langchain, openai, anthropic
import diffusers, stable_diffusion_webui

# Análisis de Datos
import pandas, polars, dask, ray
import matplotlib, seaborn, plotly, bokeh

# Especialidades
import opencv, pillow  # Computer Vision  
import spacy, nltk    # NLP
import prophet, tslearn  # Time Series
import gymnasium, stable_baselines3  # RL
import networkx, igraph  # Graph Analysis

# Librerías Experimentales/Custom
import mi_libreria_corporativa
import experimental_ai_lib  # Desde GitHub
import custom_industrial_models  # Desarrollo propio
```

#### Mecanismos de Gestión de Dependencias

**Kepler proporciona múltiples estrategias:**

1. **requirements.txt por proyecto**:
```bash
kepler init mi-proyecto --requirements-file
# Genera requirements.txt personalizable
```

2. **Conda environments**:
```bash  
kepler env create --conda --gpu-support
# Crea environment conda con CUDA
```

3. **Poetry para resolución avanzada**:
```bash
kepler env create --poetry --python=3.11
# Usa Poetry para dependency resolution
```

4. **Docker para casos extremos**:
```bash
kepler env create --docker --base-image=pytorch/pytorch
# Container custom para dependencias complejas
```

#### Guía para Librerías Custom

**Escenario 1: Librería GitHub experimental**
```bash
# En requirements.txt del proyecto
git+https://github.com/research-lab/experimental-ai.git@v0.1.0-alpha
```

**Escenario 2: Librería corporativa privada**
```bash
# Con SSH key configurada
git+ssh://git@github.com/company/private-ml-lib.git
```

**Escenario 3: Desarrollo local**
```bash
# Estructura del proyecto Kepler
mi-proyecto/
├── kepler.yml
├── requirements.txt  
├── custom-libs/
│   └── mi-libreria/
│       ├── setup.py
│       └── mi_libreria/
└── notebooks/

# En requirements.txt
-e ./custom-libs/mi-libreria
```

**Escenario 4: Fork personalizado**
```bash
# Fork de transformers con modificaciones
git+https://github.com/mi-usuario/transformers.git@custom-industrial-models
```

#### Garantías del Framework

1. **Aislamiento por proyecto**: Cada proyecto Kepler tiene su propio environment
2. **Reproducibilidad**: Lock files automáticos para dependency pinning  
3. **Deployment automático**: Las dependencias se empaquetan automáticamente
4. **Conflict resolution**: Detección y resolución de conflictos de versiones
5. **Documentation auto**: Documentación automática de dependencias usadas

### Casos de Uso Reales

**Científico encuentra librería experimental en GitHub:**
```python
# 1. Añade a requirements.txt
git+https://github.com/research/new-algorithm.git

# 2. Kepler reconstruye environment automáticamente  
kepler env update

# 3. Usa normalmente en notebook
import new_algorithm
model = new_algorithm.ExperimentalModel()
```

**Empresa desarrolla librería interna:**
```python
# 1. Librería en repo privado
# requirements.txt:
git+ssh://git@internal-gitlab.com/ai-team/industrial-models.git

# 2. Kepler maneja autenticación SSH
kepler env create --ssh-key ~/.ssh/company_key

# 3. Deployment incluye librería automáticamente
kepler deploy model --include-private-deps
```

## 3.2. Casos de Uso Expandidos - Más Allá de Datos Industriales

### Filosofía: "Cualquier dato en Splunk, cualquier caso de uso"

Kepler está diseñado para trabajar con **cualquier tipo de datos** almacenados en Splunk, no solo datos industriales:

#### Sectores y Casos de Uso Soportados

**🏭 Industrial & Manufacturing**
- Análisis predictivo de sensores IoT
- Detección de anomalías en líneas de producción
- Optimización de procesos manufactureros
- Mantenimiento predictivo de maquinaria

**🏦 Servicios Financieros**  
- Detección de fraude en transacciones
- Análisis de riesgo crediticio
- Trading algorítmico con ML
- Compliance y auditoría automática

**🏥 Healthcare & Pharma**
- Análisis de logs de dispositivos médicos
- Detección de patrones en datos de pacientes
- Optimización de operaciones hospitalarias
- Drug discovery con IA generativa

**🛒 E-commerce & Retail**
- Sistemas de recomendación personalizados
- Análisis de comportamiento de usuarios
- Optimización de precios dinámicos
- Detección de patrones de compra

**📱 Technology & SaaS**
- Análisis de performance de aplicaciones
- Detección de anomalías en logs de sistema
- Optimización de experiencia de usuario
- Chatbots con IA generativa para soporte

**🎮 Gaming & Entertainment**
- Análisis de comportamiento de jugadores
- Sistemas de recomendación de contenido
- Detección de cheating y fraud
- Personalización de experiencias

**🚛 Logistics & Supply Chain**
- Optimización de rutas de entrega
- Predicción de demanda
- Análisis de cadena de suministro
- Tracking inteligente de inventarios

**🏛️ Government & Public Sector**
- Análisis de seguridad pública
- Optimización de servicios ciudadanos
- Detección de patrones en datos demográficos
- Smart city analytics

#### Tipos de Datos Soportados en Splunk

**📊 Datos Estructurados**
```python
# Transacciones financieras
data = kp.data.from_splunk("search index=transactions sourcetype=payment_logs")

# Eventos de aplicación web
data = kp.data.from_splunk("search index=web_logs status>=400")

# Métricas de performance
data = kp.data.from_splunk("| mstats avg(response_time) WHERE index=app_metrics")
```

**📝 Datos Semi-estructurados**
```python
# Logs de aplicación JSON
data = kp.data.from_splunk("search index=app_logs | spath")

# APIs REST logs
data = kp.data.from_splunk("search index=api_logs method=POST")
```

**📄 Datos No Estructurados**
```python
# Logs de texto libre
data = kp.data.from_splunk("search index=system_logs ERROR")

# Logs de chat/soporte
data = kp.data.from_splunk("search index=support_chats")
```

**📈 Series Temporales**
```python
# Métricas de negocio
data = kp.data.from_splunk("| mstats avg(sales_amount) WHERE index=business_metrics span=1h")

# KPIs operacionales
data = kp.data.from_splunk("| mstats max(cpu_usage) WHERE index=infrastructure span=5m")
```

#### Ejemplos de Proyectos Reales

**Proyecto E-commerce: Sistema de Recomendaciones**
```python
import kepler as kp
from transformers import AutoModel

# Extraer datos de comportamiento de usuarios
user_behavior = kp.data.from_splunk("""
search index=clickstream sourcetype=user_events
| stats count by user_id, product_category, action
""")

# Entrenar modelo de recomendaciones con transformers
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
recommendations = kp.train.custom_model(user_behavior, model)

# Desplegar API de recomendaciones
kp.deploy.to_cloud_run(recommendations, name="product-recommendations")
```

**Proyecto Healthcare: Análisis de Dispositivos Médicos**
```python
# Extraer telemetría de dispositivos médicos
device_data = kp.data.from_splunk("""
| mstats avg(heart_rate), avg(blood_pressure) 
WHERE index=medical_devices span=1m
""")

# Detección de anomalías con isolation forest
from sklearn.ensemble import IsolationForest
anomaly_model = kp.train.sklearn(device_data, algorithm="IsolationForest")

# Alertas automáticas a Splunk
kp.results.to_splunk(anomaly_model.predictions, index="medical_alerts")
```

**Proyecto Financial: Detección de Fraude**
```python
# Extraer transacciones sospechosas
transactions = kp.data.from_splunk("""
search index=payments amount>10000 OR velocity>threshold
""")

# Modelo de detección con XGBoost
fraud_model = kp.train.xgboost(transactions, target="is_fraud")

# Scoring en tiempo real
kp.deploy.real_time_scoring(fraud_model, splunk_index="fraud_scores")
```

### Arquitectura API-First: "Zero Context Switching"

**Principio Fundamental:** El científico/analista **nunca debe salir** de su entorno de trabajo (Jupyter, VSCode, Cursor). Todas las operaciones se realizan vía APIs nativas encapsuladas por Kepler.

**Stack Tecnológico Completo (API-First):**

**Data & Analytics:**
- **Splunk**: `splunk-sdk` + REST API + HEC API
- **Databases**: `psycopg2`, `pymongo`, `sqlalchemy`
- **Streaming**: `kafka-python`, `pulsar-client`

**Cloud Platforms:**
- **GCP**: `google-cloud-*` client libraries  
- **Azure**: `azure-sdk-for-python` libraries
- **AWS**: `boto3` + service-specific SDKs

**MLOps & Experiment Tracking:**
- **MLflow**: `mlflow` SDK para tracking, registry, serving
- **Weights & Biases**: `wandb` para experiment tracking avanzado
- **DVC**: `dvc` para data versioning y pipeline management
- **Great Expectations**: `great_expectations` para data quality

**Model Serving & Deployment:**
- **FastAPI**: Para crear APIs de modelos automáticamente
- **Docker**: `docker-py` para containerización automática
- **Kubernetes**: `kubernetes` client para orchestration

**Monitoring & Observability:**
- **Prometheus**: `prometheus_client` para metrics collection
- **Grafana**: `grafana-api` para dashboard automation
- **DataDog**: `datadog` para APM y monitoring

**Edge Computing:**
- **Barbara IoT**: SDK nativo para edge deployment
- **Splunk Edge Hub**: API para edge data processing
- **NVIDIA Jetson**: `jetson-inference` para edge AI

**Development & Productivity:**
- **Jupyter**: `jupyterlab` integration para notebooks
- **Git**: `pygit2` para version control automation
- **CI/CD**: GitHub Actions, GitLab CI APIs

**Evolución de Integración:**
- **v0.1-v1.0:** Splunk SDK + GCP Client Libraries
- **v1.5:** + Azure SDK (prioridad después de GCP)
- **v2.0:** + AWS SDKs (boto3)
- **v2.5:** + Edge APIs (Barbara IoT + Splunk Edge Hub)

## 4. Functional Requirements

### 4.1 Conectividad y Extracción de Datos
1. El sistema DEBE conectarse a Splunk Enterprise via REST API (puerto 8089)
2. El sistema DEBE soportar queries SPL personalizadas completas
3. El sistema DEBE extraer tanto eventos como métricas de índices Splunk
4. El sistema DEBE soportar rangos de tiempo flexibles (earliest/latest)
5. El sistema DEBE manejar automáticamente la paginación de grandes datasets
6. El sistema DEBE capturar y mostrar errores específicos de Splunk al usuario

### 4.2 Experimentación y Entrenamiento ML
7. El sistema DEBE soportar importación de cualquier librería Python estándar
8. El sistema DEBE proporcionar wrappers unificados para sklearn, XGBoost, PyTorch, TensorFlow
9. El sistema DEBE permitir entrenamiento de modelos con una API simple (`kp.train.algorithm()`)
10. El sistema DEBE serializar automáticamente modelos entrenados
11. El sistema DEBE proporcionar métricas de performance automáticas
12. El sistema DEBE soportar comparación automática entre modelos

### 4.3 Gestión de Ecosistemas
13. El sistema DEBE crear automáticamente entornos de desarrollo aislados
14. El sistema DEBE gestionar dependencias por proyecto automáticamente  
15. El sistema DEBE proporcionar templates de configuración por tipo de proyecto
16. El sistema DEBE separar completamente configuración de desarrollo/staging/producción
17. El sistema DEBE provisionar recursos GCP automáticamente por entorno

### 4.4 Despliegue y Producción
18. El sistema DEBE desplegar modelos a Google Cloud Run automáticamente
19. El sistema DEBE generar APIs REST para inferencias automáticamente
20. El sistema DEBE configurar auto-scaling basado en demanda
21. El sistema DEBE escribir resultados de predicciones automáticamente a Splunk HEC
22. El sistema DEBE crear dashboards de monitoreo automáticamente

### 4.5 Configuración y Seguridad
23. El sistema DEBE mantener credenciales fuera de repositorios git
24. El sistema DEBE soportar configuración jerárquica (global/proyecto/entorno)
25. El sistema DEBE validar conectividad y permisos automáticamente
26. El sistema DEBE rotar credenciales automáticamente
27. El sistema DEBE encriptar datos sensibles en tránsito y reposo

### 4.6 Experiencia de Usuario
28. El sistema DEBE proporcionar CLI para automatización (`kepler command`)
29. El sistema DEBE proporcionar SDK Python para notebooks (`import kepler as kp`)
30. El sistema DEBE integrarse nativamente con Jupyter notebooks
31. El sistema DEBE proporcionar autocompletado y type hints
32. El sistema DEBE mostrar mensajes de error claros y accionables
33. El sistema DEBE proporcionar logging configurable por nivel

### 4.7 Extensibilidad
34. El sistema DEBE soportar plugins dinámicos sin modificar core
35. El sistema DEBE proporcionar API para adaptadores personalizados
36. El sistema DEBE soportar múltiples clouds (GCP, AWS, Azure)
37. El sistema DEBE permitir deployment en edge devices
38. El sistema DEBE proporcionar hooks para integraciones externas

## 5. Non-Goals (Out of Scope)

### Fuera de Alcance - Fase 1
- **Conectores a bases de datos** que no sean Splunk (MySQL, PostgreSQL, etc.)
- **Interfaz gráfica web** - solo CLI y SDK Python
- **Sistemas de autenticación propios** - usar credenciales existentes
- **Gestión de usuarios y roles** - usar permisos Splunk/GCP existentes
- **Data lakes propios** - Splunk como única fuente de datos inicialmente

### Fuera de Alcance - Permanente  
- **Reemplazo completo de Splunk** - Kepler complementa, no reemplaza
- **Gestión de infraestructura física** - solo cloud y edge
- **Compliance específico por industria** - responsabilidad del usuario
- **Soporte para versiones legacy** de Python (<3.8)

## 6. Design Considerations

### Arquitectura Modular
- **Core Engine**: Gestión de configuración, logging, validaciones
- **Connectors**: Adaptadores para Splunk, GCP, otros servicios
- **Trainers**: Wrappers unificados para frameworks ML
- **Deployers**: Gestores de deployment por plataforma
- **Plugin System**: Carga dinámica de extensiones

### Experiencia de Usuario
- **API Familiar**: Similar a pandas/sklearn para adopción rápida
- **Configuración Declarativa**: YAML para configuración, Python para código
- **Error Handling Inteligente**: Mensajes específicos con sugerencias de solución
- **Documentación Progresiva**: Desde quick-start hasta patrones avanzados

### Integración con Ecosistema Existente
- **Jupyter Native**: Funciona perfectamente en notebooks
- **IDE Agnostic**: Compatible con VSCode, PyCharm, Cursor AI
- **CI/CD Ready**: Comandos CLI para pipelines automatizados
- **Monitoring Integration**: Dashboards automáticos en Splunk

## 7. Technical Considerations

### Dependencias Críticas
- **splunk-sdk**: Conectividad oficial con Splunk
- **google-cloud-run**: Deployment automático GCP
- **pandas/numpy**: Manipulación de datos estándar
- **scikit-learn/xgboost**: Frameworks ML básicos
- **pydantic**: Validación de configuración
- **typer**: CLI moderna y user-friendly

### Constraints Técnicos
- **Python 3.8+**: Requerimiento mínimo para type hints modernos
- **Splunk Enterprise**: No Splunk Cloud inicialmente (diferentes APIs)
- **GCP Billing**: Usuario debe tener proyecto GCP con billing activo
- **Network Access**: Puertos 8089 (Splunk REST) y 8088 (HEC) accesibles

### Consideraciones de Performance
- **Streaming Data**: Procesamiento incremental para datasets grandes
- **Caching Inteligente**: Cache de queries frecuentes
- **Parallel Processing**: Entrenamiento de modelos en paralelo
- **Resource Management**: Limits automáticos para evitar costos excesivos

## 8. Success Metrics

### Métricas de Adopción
- **Usuarios Activos**: >50 científicos/analistas usando Kepler mensualmente
- **Proyectos Creados**: >100 proyectos `kepler init` ejecutados
- **Modelos Desplegados**: >20 modelos en producción via Kepler
- **Retention Rate**: >80% usuarios regresan después de primer uso

### Métricas de Productividad  
- **Time-to-Production**: <1 día promedio (vs 2-4 semanas actual)
- **Development Velocity**: 3x más rápido desarrollo de modelos
- **Deployment Success Rate**: >95% deployments exitosos
- **Learning Curve**: <2 horas para primer modelo funcional

### Métricas de Costos
- **Reducción Costos Splunk**: >40% menos compute en Splunk
- **ROI Infrastructure**: Costos GCP < 60% costos Splunk evitados
- **Support Tickets**: 50% menos tickets relacionados con ML
- **Training Costs**: 70% menos tiempo en training de herramientas

### Métricas Técnicas
- **Framework Compatibility**: 100% soporte top 5 frameworks ML
- **API Uptime**: >99% disponibilidad APIs desplegadas
- **Error Rate**: <5% tasa de errores en operaciones
- **Performance**: <5s tiempo respuesta predicciones

### Métricas de Calidad
- **Code Coverage**: >85% cobertura tests automatizados
- **Documentation Score**: >4/5 utilidad documentación
- **User Satisfaction**: NPS >7/10
- **Bug Resolution**: <24h resolución bugs críticos

### Métricas de Ecosistema
- **Plugin Adoption**: >10 plugins desarrollados por comunidad
- **Integration Points**: >5 integraciones con herramientas externas
- **Multi-Cloud Usage**: >20% usuarios usando múltiples clouds
- **Edge Deployments**: >5 deployments en edge devices

## 9. Open Questions

### Preguntas Técnicas
1. **¿Cómo manejar modelos que requieren GPUs?** - Integración con Vertex AI vs Cloud Run
2. **¿Qué estrategia para versionado de modelos?** - MLflow Registry vs solución custom
3. **¿Cómo garantizar reproducibilidad?** - Docker containers vs environment pinning
4. **¿Soporte para streaming ML?** - Predicciones en tiempo real vs batch

### Preguntas de Producto
5. **¿Pricing model para cloud resources?** - Usuario paga directo vs billing centralizado  
6. **¿Soporte multi-tenant?** - Proyectos compartidos vs aislamiento completo
7. **¿Integración con sistemas legacy?** - Backwards compatibility vs modernización
8. **¿Certificaciones de seguridad?** - SOC2, ISO27001 requirements

### Preguntas de Negocio
9. **¿Modelo de licenciamiento?** - Open source vs freemium vs enterprise
10. **¿Soporte comercial?** - Community support vs paid support tiers
11. **¿Partnership strategy?** - Integración con vendors vs desarrollo independiente
12. **¿Go-to-market approach?** - Developer-first vs enterprise sales

### Preguntas de Roadmap
13. **¿Cuándo migrar de MVP a framework completo?** - Métricas de activación
14. **¿Orden de prioridad para nuevos clouds?** - AWS vs Azure después de GCP
15. **¿Cuándo introducir UI web?** - CLI/SDK first vs GUI parallel
16. **¿Estrategia de backwards compatibility?** - Breaking changes policy

---

## Estado Actual del Proyecto (6 de Septiembre de 2025)

### Contexto Temporal
**Fecha actual:** 6 de Septiembre de 2025  
**Zona horaria:** America/Bogotá  
**Fase actual según roadmap:** Fase 1 - Core ML Training (Septiembre-Octubre 2025)  
**Tiempo transcurrido desde inicio:** ~1 mes desde PRD fundacional

### Lo Que Ya Funciona (70% Alineado con Visión)
- **Conectividad Splunk bidireccional**: REST API + HEC validados
- **SDK Python nativo**: `import kepler as kp` funcionando
- **CLI funcional**: Comandos básicos implementados  
- **Integración Jupyter**: Notebooks limpios y profesionales
- **Configuración segura**: Credenciales fuera de git
- **Error handling robusto**: Errores Splunk capturados
- **Datos reales validados**: 2,890 eventos + 16 métricas extraídos

### Gaps Identificados (Próximos Pasos)
- **Model training**: Solo extracción, falta entrenamiento
- **ML frameworks support**: Solo pandas/matplotlib, faltan sklearn/PyTorch
- **Deployment automation**: Falta Cloud Run integration
- **Ecosystem management**: Falta gestión automática de entornos
- **Plugin system**: Falta extensibilidad dinámica
- **Multi-cloud**: Solo GCP, falta AWS/Azure

### Roadmap de Implementación

#### Fase 1 (Septiembre-Octubre 2025): Core ML Training
1. Implementar `kepler.train` module (sklearn + XGBoost básico)
2. Validar primer modelo end-to-end con datos reales
3. Crear sistema de serialización y versionado de modelos
4. Documentar patrones de entrenamiento

#### Fase 2 (Octubre-Noviembre 2025): Ecosystem Management  
1. Crear `kepler env` commands para gestión de entornos
2. Implementar templates de configuración por proyecto
3. Automatizar provisioning de recursos GCP
4. Separación completa dev/staging/prod

#### Fase 3 (Noviembre-Diciembre 2025): Deployment Automation
1. Implementar `kepler.deploy` module para Cloud Run
2. Crear APIs REST automáticas para modelos
3. Configurar auto-scaling y monitoring
4. Pipeline automático de resultados a Splunk

#### Fase 4 (Diciembre 2025-Enero 2026): Azure Cloud Expansion
1. **Azure Support** (Prioridad #1 después de GCP):
   - **Data sources**: Blob Storage, SQL Database, Cosmos DB, Synapse Analytics
   - **Compute**: Azure Functions, Container Instances, Azure ML Compute
   - **Deployment**: Azure ML, AKS, Azure Container Apps
   - **Monitoring**: Azure Monitor, Application Insights
   - **APIs**: `azure-sdk-for-python` complete integration

2. **MLOps Stack Integration**:
   - **MLflow**: Complete tracking, registry, serving
   - **FastAPI**: Automatic model API generation
   - **Docker**: Universal containerization
   - **Prometheus**: Metrics collection standard

#### Fase 5 (Enero-Marzo 2026): AWS Support
1. **AWS Support** (Después de Azure):
   - **Data sources**: S3, RDS, Redshift, DynamoDB
   - **Compute**: Lambda, ECS, SageMaker
   - **Deployment**: AWS Batch, EKS, AWS App Runner
   - **APIs**: `boto3` + service-specific SDKs

#### Fase 6 (Marzo-Mayo 2026): Edge Computing Expansion
1. **Barbara IoT Integration** (Prioridad #1 para Edge):
   - **Barbara IoT SDK**: Native integration para edge deployment
   - **Edge ML**: Model optimization para edge devices
   - **Offline capabilities**: Sync cuando hay conectividad
   - **Device management**: Fleet management desde Kepler

2. **Splunk Edge Hub Integration**:
   - **Edge data processing**: Local processing con sync a Splunk
   - **Hybrid architecture**: Edge + Cloud seamless

3. **Additional Edge Platforms**:
   - **NVIDIA Jetson**: AI inference optimization
   - **Raspberry Pi**: Lightweight deployment
   - **Industrial PCs**: Factory floor deployment

#### Fase 7 (2026): Data Sources & Ecosystem Completo
1. **Database Connectors**: PostgreSQL, MySQL, MongoDB, Cassandra
2. **API Connectors**: REST APIs, GraphQL, webhooks
3. **File Connectors**: CSV, Parquet, JSON, Excel, PDF
4. **Streaming Sources**: Kafka, Pulsar, RabbitMQ
5. **Plugin marketplace** y community contributions
6. **Enterprise features** y governance

---

**Este PRD es un documento vivo que evolucionará con el proyecto. Se actualizará trimestralmente o ante cambios significativos en requirements o arquitectura.**
