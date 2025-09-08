# Manual de Usuario - Kepler Framework

> **Versión:** 1.0  
> **Fecha:** 7 de Septiembre de 2025  
> **Audiencia:** Ingenieros de Datos, Científicos de Datos, Analistas

---

## 👨‍💼 **ROL: INGENIERO DE DATOS - SETUP INICIAL**

### **📋 Prerrequisitos**

Verificar versión de Python:
```bash
python --version
# Requerido: Python 3.10 o superior
```

Verificar Git:
```bash
git --version
# Requerido para clonar el repositorio
```

### **📦 Paso 1: Clonar el Repositorio**

```bash
# Clonar repositorio de Kepler Framework
git clone https://github.com/company/kepler-framework.git
cd kepler-framework
```

### **🔧 Paso 2: Configurar Entorno Virtual**

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En macOS/Linux:
source .venv/bin/activate
# En Windows:
# .venv\Scripts\activate
```

### **⚙️ Paso 3: Instalar Kepler Framework**

```bash
# Instalar framework en modo desarrollo
pip install -e .

# Verificar instalación
kepler --version
```

**Resultado esperado:**
```
Kepler Framework - Simple ML for Industrial Data
```

### **🚀 Paso 4: Crear Nuevo Proyecto**

```bash
# Crear proyecto de monitoreo de sensores
kepler init sensor-monitoring
```

**Resultado esperado:**
```
🎉 Kepler project 'sensor-monitoring' initialized successfully!
```

### **📁 Paso 5: Verificar Estructura del Proyecto**

```bash
# Entrar al directorio del proyecto
cd sensor-monitoring

# Verificar estructura
ls -la
```

**Estructura creada:**
```
sensor-monitoring/
├── .env.template          # Template de variables de entorno
├── .gitignore            # Archivos a ignorar en Git
├── kepler.yml            # Configuración del proyecto
├── README.md             # Documentación del proyecto
├── data/                 # Datos del proyecto
├── logs/                 # Logs de ejecución
├── models/               # Modelos entrenados
├── notebooks/            # Notebooks de análisis
└── scripts/              # Scripts personalizados
```

---

## 🔐 **CONFIGURACIÓN DE CREDENCIALES**

### **Paso 6: Configurar Variables de Entorno**

```bash
# Copiar template a archivo .env
cp .env.template .env

# Editar .env con credenciales reales
# SPLUNK_TOKEN=tu_token_rest_api_aqui
# SPLUNK_HEC_TOKEN=tu_token_hec_aqui
# GCP_PROJECT_ID=tu_proyecto_gcp_aqui
# SPLUNK_HOST=https://tu-servidor-splunk:8089
# SPLUNK_HEC_URL=http://tu-servidor-splunk:8088/services/collector
```

### **Paso 7: Validar Prerequisites**

```bash
# Validar prerequisites del sistema
kepler validate prerequisites
```

**Resultado esperado:**
```
✅ Python 3.13.4 is compatible
✅ Kepler v0.2.1 is properly installed
✅ Jupyter is available
✅ Splunk SDK is available and ready
```

### **Paso 8: Instalar Dependencias MLOps (Opcional)**

```bash
# Instalar herramientas MLOps recomendadas
pip install mlflow dvc
```

### **Paso 9: Validar Ecosistema Completo**

```bash
# Validación completa del ecosistema
kepler validate ecosystem
```

**Resultado esperado:**
```
✅ Overall Status: SUCCESS
✅ Success Rate: 100.0% (20/20)

✅ Splunk: Conectividad, autenticación, 15 índices, HEC
✅ GCP: SDK, autenticación, Cloud Run, Artifact Registry
✅ MLOps: MLflow + DVC disponibles
✅ APIs: Todas las funcionalidades listas
```

### **🔧 Troubleshooting (Si es necesario)**

**Problema común: Token expirado**
```bash
# Si falla autenticación, generar nuevo token en Splunk
# Actualizar SPLUNK_TOKEN en .env
# Volver a validar
kepler validate ecosystem
```

---

## ✅ **VERIFICACIÓN FINAL**

### **Estado del Ecosistema:**
- **Splunk**: 100% funcional (API + HEC)
- **GCP**: 100% funcional (deployment listo)
- **MLOps**: 100% funcional (versioning listo)
- **Kepler**: 100% funcional (todas las APIs)

---

## 🚀 **JUAN COMPLETÓ SETUP EXITOSAMENTE**

**El ecosistema está 100% validado y listo para uso productivo.**

### **📝 Próximos Pasos para Ana (Científica de Datos):**

1. **Crear requirements.txt** para librerías AI
2. **Extraer datos** desde Splunk
3. **Entrenar modelos** con AutoML
4. **Desplegar a producción**

---

**✅ Setup del Ingeniero completado. Ecosistema listo para científicos de datos.**
