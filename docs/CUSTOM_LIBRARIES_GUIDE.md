# Kepler Framework - Guía de Librerías Custom y Experimentales

> **Última actualización:** 6 de Septiembre de 2025  
> **Filosofía:** "Si está en Python, Kepler lo soporta"  
> **Audiencia:** Científicos de Datos, Ingenieros de IA, Investigadores

## 🎯 Filosofía del Soporte Ilimitado

Kepler está diseñado para **nunca limitar** las opciones tecnológicas del científico de datos. No importa si es una librería oficial de PyPI, un experimento de GitHub, un desarrollo corporativo interno, o tu propio algoritmo custom - **Kepler lo soporta**.

## 📦 Fuentes de Librerías Soportadas

### 1. PyPI Oficial (Estándar)
```bash
# requirements.txt
numpy==1.24.3
pandas>=2.0.0
scikit-learn
transformers[torch]
```

### 2. Repositorios GitHub/GitLab
```bash
# requirements.txt
# Versión específica
git+https://github.com/huggingface/transformers.git@v4.21.0

# Rama específica  
git+https://github.com/research-lab/experimental-ai.git@main

# Tag específico
git+https://github.com/company/ml-toolkit.git@v2.1.0-alpha
```

### 3. Repositorios Privados
```bash
# Con SSH (recomendado para repos privados)
git+ssh://git@github.com/company/private-ml-lib.git

# Con HTTPS y token
git+https://token:x-oauth-basic@github.com/company/private-repo.git
```

### 4. Librerías Locales (Desarrollo Propio)
```bash
# requirements.txt
-e ./custom-libs/mi-algoritmo
-e ./libs/industrial-models
```

### 5. Archivos Wheel/Tar Locales
```bash
# requirements.txt
./wheels/custom-library-1.0.0-py3-none-any.whl
./dist/mi-algoritmo-0.1.tar.gz
```

## 🛠️ Casos de Uso Detallados

### Caso 1: Librería Experimental de GitHub

**Escenario:** Encontraste un paper con código en GitHub que quieres probar.

```bash
# 1. Estructura del proyecto
mi-proyecto/
├── kepler.yml
├── requirements.txt
├── notebooks/
│   └── experimento.ipynb
└── data/

# 2. requirements.txt
git+https://github.com/research-lab/novel-algorithm.git@v0.1.0-alpha
pandas>=2.0.0
matplotlib

# 3. Kepler maneja la instalación automáticamente
kepler env update

# 4. Uso en notebook
import novel_algorithm
import kepler as kp

# Extraer datos
data = kp.data.from_splunk("search index=sensors")

# Usar algoritmo experimental
model = novel_algorithm.ExperimentalModel()
results = model.fit_predict(data)
```

### Caso 2: Librería Corporativa Privada

**Escenario:** Tu empresa desarrolló librerías internas de IA industrial.

```bash
# 1. Configurar SSH key para repos privados
kepler config set-ssh-key ~/.ssh/company_deploy_key

# 2. requirements.txt
git+ssh://git@internal-gitlab.company.com/ai-team/industrial-models.git
git+ssh://git@internal-gitlab.company.com/ai-team/sensor-analytics.git@v1.2.3

# 3. Kepler maneja autenticación automáticamente
kepler env create --ssh-auth

# 4. Deployment incluye librerías privadas
kepler deploy model --include-private-deps
```

### Caso 3: Desarrollo Local Custom

**Escenario:** Estás desarrollando tu propio algoritmo mientras experimentas.

```bash
# 1. Estructura del proyecto
mi-proyecto/
├── kepler.yml
├── requirements.txt
├── custom-libs/
│   └── mi-algoritmo/
│       ├── setup.py
│       ├── mi_algoritmo/
│       │   ├── __init__.py
│       │   └── core.py
│       └── tests/
├── notebooks/
└── data/

# 2. setup.py en custom-libs/mi-algoritmo/
from setuptools import setup, find_packages

setup(
    name="mi-algoritmo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn"
    ]
)

# 3. requirements.txt
-e ./custom-libs/mi-algoritmo
pandas
matplotlib

# 4. Desarrollo iterativo
kepler env update  # Reinstala automáticamente cuando cambias código
```

### Caso 4: Fork Personalizado

**Escenario:** Modificaste una librería existente para tus necesidades específicas.

```bash
# 1. Fork transformers en GitHub y modifícalo
# 2. requirements.txt
git+https://github.com/tu-usuario/transformers.git@custom-industrial-models

# 3. Kepler usa tu fork automáticamente
import transformers  # Tu versión modificada
```

### Caso 5: Múltiples Fuentes Mezcladas

**Escenario:** Proyecto complejo con librerías de múltiples fuentes.

```bash
# requirements.txt - Proyecto real complejo
# PyPI oficial
numpy==1.24.3
pandas>=2.0.0
scikit-learn

# Deep Learning
torch>=2.0.0
transformers[torch]

# Experimental de GitHub
git+https://github.com/research/novel-transformer.git@v0.2.0-beta

# Corporativo privado
git+ssh://git@gitlab.company.com/ai/industrial-nlp.git@v1.1.0

# Desarrollo local
-e ./libs/custom-preprocessing
-e ./libs/domain-specific-models

# Fork personalizado
git+https://github.com/mi-usuario/langchain.git@industrial-agents

# Wheel local (librería compilada)
./wheels/optimized-inference-1.0.0-py3-none-any.whl
```

## ⚙️ Gestión Avanzada de Dependencias

### Environments por Tipo de Proyecto

```bash
# Proyecto ML tradicional
kepler env create --template=ml-basic
# Instala: sklearn, pandas, matplotlib, jupyter

# Proyecto Deep Learning  
kepler env create --template=deep-learning --gpu
# Instala: pytorch, tensorflow, cuda-toolkit

# Proyecto IA Generativa
kepler env create --template=generative-ai
# Instala: transformers, langchain, openai, anthropic

# Proyecto custom completo
kepler env create --from-requirements=requirements.txt
```

### Resolución de Conflictos

```bash
# Kepler detecta y resuelve conflictos automáticamente
kepler env diagnose

# Output ejemplo:
# ⚠️  Conflict detected: 
#    transformers==4.21.0 (from PyPI)
#    transformers@custom-branch (from GitHub fork)
# 
# 🔧 Resolution: Using GitHub fork (more specific)
# 
# ✅ Environment ready
```

### Lock Files para Reproducibilidad

```bash
# Kepler genera lock files automáticamente
kepler env lock

# Genera:
# kepler-lock.txt - Versiones exactas instaladas
# kepler-sources.txt - URLs y commits exactos de repos Git
```

## 🚀 Deployment con Librerías Custom

### Empaquetado Automático

```bash
# Kepler empaqueta TODAS las dependencias automáticamente
kepler deploy model --target=cloud-run

# Incluye:
# - Librerías PyPI con versiones exactas
# - Código de repos Git (clonado y empaquetado)  
# - Librerías locales (copiadas al container)
# - Wheels locales (incluidos en imagen)
```

### Deployment con Librerías Privadas

```bash
# Para librerías privadas, Kepler usa SSH forwarding
kepler deploy model --ssh-forward --private-repos

# O crea container con credenciales embebidas (menos seguro)
kepler deploy model --embed-credentials
```

## 🔍 Debugging y Troubleshooting

### Diagnóstico de Instalación

```bash
# Ver qué librerías están instaladas y de dónde vienen
kepler env list --sources

# Output ejemplo:
# numpy==1.24.3 (PyPI)
# pandas==2.0.3 (PyPI)  
# transformers==4.21.0 (git+https://github.com/huggingface/transformers.git@v4.21.0)
# mi-algoritmo==0.1.0 (-e ./custom-libs/mi-algoritmo)
```

### Problemas Comunes y Soluciones

**Error: "Could not find a version that satisfies the requirement"**
```bash
# Solución: Verificar que el repo/branch existe
kepler env validate-requirements

# O instalar manualmente para debug
pip install git+https://github.com/repo/lib.git@branch --verbose
```

**Error: "Permission denied (publickey)"**
```bash
# Solución: Configurar SSH key
kepler config set-ssh-key ~/.ssh/id_rsa
# O usar HTTPS con token
```

**Error: "No module named 'custom_lib'"**
```bash
# Solución: Reinstalar en modo editable
kepler env update --force-reinstall
```

## 📋 Best Practices

### 1. Versionado Explícito
```bash
# ✅ Bueno - versión específica
git+https://github.com/repo/lib.git@v1.2.3

# ❌ Malo - siempre latest (no reproducible)
git+https://github.com/repo/lib.git
```

### 2. Documentación de Dependencias
```bash
# requirements.txt comentado
# Core ML
scikit-learn==1.3.0
pandas>=2.0.0

# Experimental algorithm from Research Lab
git+https://github.com/research/novel-ai.git@v0.1.0  # Paper: arxiv.org/1234.5678

# Company internal tools
git+ssh://git@internal.company.com/ai/tools.git@v2.1.0  # Internal documentation: wiki.company.com/ai-tools
```

### 3. Testing con Librerías Custom
```python
# tests/test_custom_integration.py
def test_custom_library_integration():
    """Test que la librería custom funciona con Kepler"""
    import mi_algoritmo
    import kepler as kp
    
    # Test básico de importación
    assert hasattr(mi_algoritmo, 'CustomModel')
    
    # Test de integración con datos Kepler
    data = kp.data.from_splunk("search index=test")
    model = mi_algoritmo.CustomModel()
    result = model.predict(data)
    
    assert result is not None
```

### 4. CI/CD con Librerías Custom
```yaml
# .github/workflows/kepler-test.yml
name: Test with Custom Libraries
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Kepler Environment
      run: |
        pip install kepler-framework
        kepler env create --from-requirements
    - name: Test Custom Libraries
      run: |
        kepler test --include-custom-libs
```

## 🎯 Próximos Pasos

1. **Experimenta**: Prueba con una librería experimental de GitHub
2. **Desarrolla**: Crea tu propia librería custom local  
3. **Integra**: Combina múltiples fuentes en un proyecto
4. **Despliega**: Lleva tu experimento a producción con todas las dependencias

**¿Necesitas ayuda específica?** 
- 📧 Email: support@kepler-framework.org
- 💬 Discord: https://discord.gg/kepler-framework  
- 📖 Docs: https://docs.kepler-framework.org/custom-libraries

---

**Recuerda: En Kepler, si está en Python, lo soportamos. Sin excepciones.**
