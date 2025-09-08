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

## ✅ **VERIFICACIÓN DE SETUP**

### **Comandos de Verificación:**

```bash
# Verificar comandos disponibles
kepler --help

# Verificar estructura del proyecto
pwd
# Debe mostrar: /ruta/al/proyecto/sensor-monitoring
```

---

## 📝 **Próximos Pasos**

1. **Configurar credenciales** (Splunk + GCP)
2. **Validar ecosistema** con `kepler validate`
3. **Extraer datos** desde Splunk
4. **Entrenar modelos** con AutoML
5. **Desplegar a producción**

---

**✅ Setup completado. Juan puede proceder con la configuración de credenciales.**
