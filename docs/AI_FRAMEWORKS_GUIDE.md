# Kepler Framework - Complete AI Frameworks Support Guide

> **Última actualización:** 6 de Septiembre de 2025  
> **Filosofía:** "Si está en Python, Kepler lo soporta - sin excepciones"  
> **Audiencia:** Científicos de Datos, Ingenieros de IA, Investigadores

## 🎯 Unlimited AI Framework Support

Kepler está diseñado para soportar **CUALQUIER framework de Inteligencia Artificial** disponible en Python, desde librerías oficiales de PyPI hasta desarrollos experimentales en GitHub.

### ✅ **Frameworks Soportados (Ejemplos)**

#### 🤖 **Machine Learning Tradicional**
```python
# Desde CLI
kepler libs template --template ml

# Incluye automáticamente:
import sklearn        # Scikit-learn para ML clásico
import xgboost       # Gradient boosting optimizado
import lightgbm      # LightGBM para datasets grandes
import catboost      # CatBoost para datos categóricos
```

#### 🧠 **Deep Learning**
```python
# Desde CLI
kepler libs template --template deep_learning

# Incluye automáticamente:
import torch         # PyTorch para deep learning
import tensorflow    # TensorFlow/Keras
import jax          # JAX para computación científica
import lightning    # PyTorch Lightning para training
```

#### 🎨 **Inteligencia Artificial Generativa**
```python
# Desde CLI
kepler libs template --template generative_ai

# Incluye automáticamente:
import transformers  # Hugging Face Transformers (LLMs)
import langchain     # LangChain para AI agents
import openai        # OpenAI API (GPT, DALL-E)
import anthropic     # Claude API
import diffusers     # Stable Diffusion y modelos generativos
```

#### 👁️ **Computer Vision**
```python
# Desde CLI
kepler libs template --template computer_vision

# Incluye automáticamente:
import cv2           # OpenCV para procesamiento de imágenes
from PIL import Image  # Pillow para manipulación de imágenes
import torchvision   # PyTorch Vision para CV
import albumentations  # Augmentations avanzadas
```

#### 📝 **Natural Language Processing**
```python
# Desde CLI
kepler libs template --template nlp

# Incluye automáticamente:
import spacy         # spaCy para NLP avanzado
import nltk          # NLTK para procesamiento de texto
import transformers  # Transformers para NLP moderno
import datasets      # Datasets para entrenamiento
```

#### 🌟 **Ecosistema Completo de IA**
```python
# Desde CLI
kepler libs template --template full_ai

# Incluye TODAS las categorías anteriores:
# ML + Deep Learning + Generative AI + Computer Vision + NLP
```

## 🛠️ **Gestión Avanzada de Librerías**

### **Comandos CLI Disponibles**

```bash
# Crear template de IA específico
kepler libs template --template generative_ai
kepler libs template --template deep_learning
kepler libs template --template computer_vision

# Instalar librería específica
kepler libs install --library transformers
kepler libs install --library "torch>=2.0.0"

# Instalar desde GitHub experimental
kepler libs install --library git+https://github.com/research/experimental-ai.git

# Instalar todas las librerías del proyecto
kepler libs install

# Listar librerías instaladas
kepler libs list

# Validar entorno de librerías
kepler libs validate
```

### **API Python (SDK)**

```python
import kepler as kp

# Crear template programáticamente
kp.libs.create_template("generative_ai")

# Instalar librerías programáticamente  
kp.libs.install("transformers>=4.30.0")
kp.libs.install("git+https://github.com/research/novel-ai.git")

# Validar entorno
report = kp.libs.validate_environment()
print(f"Libraries: {report['successful_imports']}/{report['total_libraries']}")
```

## 🔬 **Casos de Uso Reales**

### **Caso 1: Proyecto de IA Generativa para Análisis de Logs**

```python
# 1. Crear proyecto con template de IA Generativa
kepler init log-analysis-ai
cd log-analysis-ai
kepler libs template --template generative_ai

# 2. Instalar librerías automáticamente
kepler libs install

# 3. Usar en notebook
import kepler as kp
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import langchain

# Extraer logs de Splunk
logs_data = kp.data.from_splunk("search index=system_logs ERROR")

# Análisis con LLMs
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

# Procesar con LangChain
from langchain.chains import LLMChain
chain = LLMChain(llm=model, prompt="Analyze this log entry: {log}")

# Desplegar automáticamente
kp.deploy.to_cloud_run(model, name="log-analyzer")
```

### **Caso 2: Computer Vision para Análisis Industrial**

```python
# 1. Template específico de Computer Vision
kepler libs template --template computer_vision

# 2. Librerías especializadas adicionales
kepler libs install --library "opencv-contrib-python>=4.8.0"
kepler libs install --library git+https://github.com/roboflow/supervision.git

# 3. Desarrollo
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Extraer datos de imágenes desde Splunk
image_metadata = kp.data.from_splunk("search index=camera_logs")

# Procesamiento con OpenCV
def analyze_industrial_image(image_path):
    img = cv2.imread(image_path)
    # ... procesamiento específico
    return results

# Desplegar pipeline completo
kp.deploy.to_cloud_run(analyze_industrial_image, name="vision-analyzer")
```

### **Caso 3: Librería Experimental de GitHub**

```python
# 1. Encontrar librería experimental en paper de investigación
# Paper: "Novel Transformer Architecture for Time Series"
# Repo: https://github.com/research-lab/novel-transformer

# 2. Instalar directamente desde GitHub
kepler libs install --library git+https://github.com/research-lab/novel-transformer.git@v0.1.0-alpha

# 3. Usar inmediatamente
import novel_transformer
import kepler as kp

# Datos de series temporales desde Splunk
timeseries_data = kp.data.from_splunk("| mstats avg(cpu_usage) WHERE index=metrics span=5m")

# Experimentar con algoritmo novel
model = novel_transformer.NovelTransformer(
    input_size=timeseries_data.shape[1],
    hidden_size=128
)

# Entrenar y desplegar
trained_model = kp.train.custom_model(timeseries_data, model)
kp.deploy.to_cloud_run(trained_model, name="novel-timeseries")
```

### **Caso 4: Desarrollo Custom Corporativo**

```python
# 1. Estructura de proyecto con librería interna
my-ai-project/
├── kepler.yml
├── requirements.txt
├── custom-libs/
│   └── company-ai-toolkit/
│       ├── setup.py
│       └── company_ai/
│           ├── __init__.py
│           └── proprietary_algorithms.py
└── notebooks/

# 2. requirements.txt
"""
# Librerías estándar
transformers>=4.30.0
torch>=2.0.0

# Librería corporativa local
-e ./custom-libs/company-ai-toolkit

# Librería corporativa privada
git+ssh://git@internal-gitlab.company.com/ai-team/advanced-models.git@v2.1.0
"""

# 3. Instalación automática
kepler libs install

# 4. Uso en producción
import company_ai
import advanced_models
import kepler as kp

# Combinar librerías corporativas con estándar
data = kp.data.from_splunk("search index=business_data")
model = company_ai.ProprietaryModel(advanced_models.CustomOptimizer())
result = kp.train.custom_model(data, model)
```

## 🚀 **Deployment con Librerías Custom**

### **Empaquetado Automático**

Kepler empaqueta automáticamente TODAS las dependencias:

```python
# Deployment incluye automáticamente:
# - Librerías PyPI con versiones exactas
# - Código de repositorios Git (clonado)
# - Librerías locales custom (copiadas)
# - Wheels compiladas (incluidas)

kp.deploy.to_cloud_run(
    model, 
    name="custom-ai-model",
    include_all_dependencies=True,  # Default: True
    optimize_for_inference=True     # Optimiza container para inference
)
```

### **Deployment Multi-Cloud con Librerías Custom**

```python
# Mismo modelo, múltiples clouds, todas las dependencias
kp.deploy.to_gcp(model, name="ai-model-gcp")
kp.deploy.to_azure(model, name="ai-model-azure")  # Future
kp.deploy.to_barbara_iot(model, device="edge-001")  # Future
```

## 🔍 **Troubleshooting y Debugging**

### **Diagnóstico de Librerías**

```bash
# Ver todas las librerías instaladas
kepler libs list

# Validar entorno completo
kepler libs validate

# Información detallada de una librería
kepler libs info --library transformers
```

### **Resolución de Conflictos**

```python
# Kepler detecta y resuelve conflictos automáticamente
kepler libs diagnose

# Output ejemplo:
# ⚠️  Conflict detected: 
#    transformers==4.21.0 (from PyPI)
#    transformers@custom-branch (from GitHub fork)
# 
# 🔧 Resolution: Using GitHub fork (more specific)
# ✅ Environment ready
```

### **Problemas Comunes**

**Error: "Could not find a version that satisfies the requirement"**
```bash
# Verificar que el repo/branch existe
kepler libs validate --library git+https://github.com/user/repo.git@branch

# O instalar manualmente para debug
pip install git+https://github.com/user/repo.git@branch --verbose
```

**Error: "Permission denied (publickey)"**
```bash
# Configurar SSH key para repos privados
kepler setup ssh-key ~/.ssh/company_key

# O usar HTTPS con token
git config --global url."https://token:x-oauth-basic@github.com/".insteadOf "git@github.com:"
```

## 📋 **Best Practices**

### **1. Versionado Explícito**
```python
# ✅ Bueno - versión específica para reproducibilidad
transformers>=4.30.0,<5.0.0
git+https://github.com/research/ai-lib.git@v1.2.3

# ❌ Malo - siempre latest (no reproducible)
transformers
git+https://github.com/research/ai-lib.git
```

### **2. Templates por Tipo de Proyecto**
```bash
# ✅ Empezar con template apropiado
kepler libs template --template generative_ai  # Para proyectos LLM
kepler libs template --template computer_vision  # Para proyectos CV
kepler libs template --template deep_learning   # Para proyectos DL

# Luego añadir librerías específicas
kepler libs install --library "custom-research-lib>=1.0.0"
```

### **3. Documentación de Dependencias**
```python
# requirements.txt bien documentado
# === GENERATIVE AI FRAMEWORKS ===
transformers>=4.30.0  # Hugging Face Transformers para LLMs
langchain>=0.0.200    # LangChain para AI agents
openai>=0.27.0        # OpenAI API para GPT/DALL-E

# === RESEARCH LIBRARIES ===
git+https://github.com/research/novel-ai.git@v0.1.0  # Paper: arxiv.org/1234.5678

# === CORPORATE TOOLS ===
git+ssh://git@github.com/company/ai-toolkit.git@v2.1.0  # Internal docs: wiki.company.com
```

## 🎯 **Próximos Pasos**

1. **Experimenta**: Crea un proyecto con template de IA Generativa
2. **Personaliza**: Añade librerías experimentales de GitHub
3. **Desarrolla**: Combina múltiples frameworks en un proyecto
4. **Despliega**: Lleva tu experimento a producción con todas las dependencias

**Recuerda:** En Kepler, cualquier librería Python es soportada. **Sin limitaciones, sin excepciones.**

---

**¿Necesitas ayuda específica?**
- 📧 Soporte: Documentación completa en docs/
- 💬 Comunidad: Comparte tus experimentos con librerías custom
- 🔧 Desarrollo: Contribuye con nuevos templates de IA
