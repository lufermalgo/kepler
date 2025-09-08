# Kepler Framework - Guía de Validación y Troubleshooting

> **Última actualización:** 7 de Septiembre de 2025  
> **Estado:** Completamente Implementado (M4 - Tasks 7.1-7.10)  
> **Audiencia:** DevOps, Ingenieros, Administradores de Sistema

## 🎯 Sistema de Validación Completa

Kepler incluye un sistema de validación comprehensivo que verifica todo el ecosistema antes de trabajar, proporcionando mensajes accionables y fixes automáticos.

### **Filosofía: "Validar todo antes de empezar a trabajar"**

El sistema de validación verifica:
- ✅ **Prerequisites**: Python, librerías, Jupyter
- ✅ **Splunk**: Conectividad, autenticación, índices, HEC
- ✅ **GCP**: CLI, autenticación, APIs, permisos
- ✅ **MLOps**: MLflow, DVC (opcional)
- ✅ **End-to-End**: Workflows completos

---

## 🚀 Comandos CLI

### **Validación Completa del Ecosistema**

```bash
# Validación completa
kepler validate ecosystem

# Con auto-fixes automáticos
kepler validate ecosystem --auto-fix

# Solo componentes críticos (sin MLflow/DVC)
kepler validate ecosystem --skip-optional

# Salida en JSON para CI/CD
kepler validate ecosystem --format json --save validation-report.json
```

### **Validación por Plataforma**

```bash
# Solo Splunk
kepler validate splunk

# Solo GCP
kepler validate gcp

# Solo prerequisites
kepler validate prerequisites
```

### **Setup Guiado de Plataformas**

```bash
# Setup interactivo de Splunk
kepler setup splunk

# Setup interactivo de GCP
kepler setup gcp

# Setup con almacenamiento seguro de credenciales
kepler setup splunk --secure

# Setup sin validación posterior
kepler setup gcp --no-validate
```

### **Diagnóstico Inteligente**

```bash
# Auto-detección de problemas
kepler diagnose

# Problemas de conectividad
kepler diagnose connection

# Problemas de autenticación
kepler diagnose authentication --platform splunk

# Problemas de deployment
kepler diagnose deployment --verbose
```

---

## 🔍 API de Validación (SDK)

### **Validación Programática**

```python
import kepler as kp

# Validación completa del ecosistema
report = kp.validate.ecosystem()

print(f"Estado general: {report.overall_status.value}")
print(f"Tasa de éxito: {report.success_rate:.1f}%")

# Mostrar recomendaciones
for recommendation in report.recommendations:
    print(f"💡 {recommendation}")

# Verificar si hay problemas críticos
critical_issues = [r for r in report.results if r.level.value == "critical" and not r.success]
if critical_issues:
    print("🚨 Problemas críticos encontrados:")
    for issue in critical_issues:
        print(f"   • {issue.check_name}: {issue.hint}")
```

### **Validación por Plataforma**

```python
# Validar solo Splunk
splunk_results = kp.validate.splunk()
for result in splunk_results:
    if not result.success:
        print(f"❌ {result.check_name}: {result.message}")
        print(f"💡 {result.hint}")

# Validar solo GCP
gcp_results = kp.validate.gcp()
gcp_healthy = all(r.success for r in gcp_results)
print(f"GCP Status: {'✅ Healthy' if gcp_healthy else '❌ Issues detected'}")
```

---

## 🔐 Gestión Segura de Credenciales

### **Almacenamiento Seguro**

```python
import kepler as kp

# Almacenar credenciales con encriptación AES-256
kp.security.store_credential("splunk_token", "your-secret-token")
kp.security.store_credential("gcp_service_account", "service-account-json")

# Listar credenciales almacenadas (sin mostrar valores)
credentials = kp.security.list_credentials()
for cred in credentials:
    print(f"{cred.name}: {cred.source} ({'encrypted' if cred.encrypted else 'plain'})")
```

### **Recuperación Segura**

```python
# Recuperar credenciales con fallback a variables de entorno
splunk_token = kp.security.get_credential("splunk_token", "SPLUNK_TOKEN")
gcp_key = kp.security.get_credential("gcp_key", "GOOGLE_APPLICATION_CREDENTIALS")

# Validar postura de seguridad
security_status = kp.security.validate_security()
if security_status["overall_secure"]:
    print("✅ Postura de seguridad es buena")
else:
    print("⚠️ Problemas de seguridad detectados:")
    for issue in security_status["issues"]:
        print(f"   • {issue['message']}")
```

---

## 🛠️ Troubleshooting Common Issues

### **Problemas de Conectividad Splunk**

**Error:** `Splunk server returned HTTP 401`
```bash
# Diagnóstico
kepler diagnose authentication --platform splunk

# Solución
kepler setup splunk  # Re-configurar token
```

**Error:** `SSL certificate verification failed`
```bash
# Fix automático
kepler validate splunk --auto-fix

# O manual
kepler setup splunk  # Configurar verify_ssl: false
```

### **Problemas de Autenticación GCP**

**Error:** `No active GCP authentication`
```bash
# Solución
gcloud auth login
kepler validate gcp
```

**Error:** `Cloud Run API not enabled`
```bash
# Fix automático
kepler validate gcp --auto-fix

# O manual
gcloud services enable run.googleapis.com
```

### **Problemas de Librerías**

**Error:** `Library environment has issues (60% success rate)`
```bash
# Diagnóstico detallado
kepler diagnose --verbose

# Fix automático
kepler libs install
```

---

## 📊 Interpretación de Resultados

### **Niveles de Validación**

- **✅ SUCCESS**: Todo funciona correctamente
- **⚠️ WARNING**: Funcionalidad reducida, pero Kepler funciona
- **❌ CRITICAL**: Bloquea funcionalidad, debe corregirse

### **Categorías de Validación**

- **Prerequisites**: Python, librerías básicas, herramientas
- **Authentication**: Tokens, credenciales, permisos
- **Connectivity**: Red, APIs, endpoints
- **Configuration**: Archivos config, settings
- **Permissions**: Accesos, roles, políticas
- **Functionality**: Workflows end-to-end

### **Ejemplo de Reporte**

```
🔍 Kepler Ecosystem Validation Report
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Check                ┃ Status      ┃ Message                          ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python version       │ ✅ success  │ Python 3.11.5 is compatible     │
│ Kepler installation  │ ✅ success  │ Kepler v0.1.0 properly installed│
│ Splunk connectivity  │ ✅ success  │ Splunk server is accessible     │
│ GCP authentication   │ ❌ critical │ No active GCP authentication    │
│ Cloud Run API        │ ❌ critical │ Cloud Run API not enabled       │
└──────────────────────┴─────────────┴──────────────────────────────────┘

❌ Overall Status: CRITICAL
Success Rate: 60.0% (3/5)

📋 Recommendations:
🚨 CRITICAL: Fix these issues before using Kepler:
   • GCP authentication: Run: gcloud auth login
   • Cloud Run API: Enable with: gcloud services enable run.googleapis.com

🔧 AUTO-FIXES: Run these commands to fix issues automatically:
   • gcloud auth login
   • gcloud services enable run.googleapis.com
```

---

## 🔧 Configuración Avanzada

### **Configuración Personalizada**

```yaml
# ~/.kepler/config.yml
splunk:
  host: "https://splunk.company.com:8089"
  token: "stored_securely"  # Almacenado con AES-256
  hec_token: "stored_securely"
  verify_ssl: true
  timeout: 30

gcp:
  project_id: "my-ml-project"
  region: "us-central1"
  apis_enabled:
    - "run.googleapis.com"
    - "cloudbuild.googleapis.com"
    - "artifactregistry.googleapis.com"

validation:
  auto_fix_enabled: true
  include_optional_tools: true
  timeout_seconds: 60
```

### **Variables de Entorno**

```bash
# Credenciales (fallback si no están almacenadas securely)
export SPLUNK_TOKEN="your-splunk-token"
export SPLUNK_HEC_TOKEN="your-hec-token"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Configuración de validación
export KEPLER_VALIDATION_TIMEOUT=120
export KEPLER_AUTO_FIX=true
```

---

## 🚨 Guías de Emergencia

### **Kepler No Funciona - Lista de Verificación**

1. **Verificar Prerequisites**:
```bash
python --version  # Debe ser 3.8+
kepler --version  # Debe mostrar versión
kepler validate prerequisites
```

2. **Verificar Configuración**:
```bash
ls ~/.kepler/config.yml  # Debe existir
kepler validate ecosystem --format summary
```

3. **Verificar Conectividad**:
```bash
kepler validate splunk
kepler validate gcp
```

4. **Fix Automático**:
```bash
kepler validate ecosystem --auto-fix
```

### **Deployment Falla - Diagnóstico**

1. **Verificar GCP Setup**:
```bash
gcloud auth list
gcloud config get-value project
kepler validate gcp
```

2. **Verificar Modelo**:
```bash
python -c "import joblib; model = joblib.load('model.pkl'); print(type(model))"
```

3. **Deployment con Diagnóstico**:
```bash
kepler deploy model.pkl --dry-run  # Ver plan
kepler diagnose deployment --verbose
```

---

## 📋 Códigos de Error Estándar

### **Validation Errors (VALIDATE_001)**
- Falla en validación de prerequisites
- **Fix**: `kepler validate prerequisites --auto-fix`

### **Security Errors (SECURITY_001-002)**
- SECURITY_001: Master password required
- SECURITY_002: Credential storage failed
- **Fix**: Configurar almacenamiento seguro

### **Platform Errors**
- SPLUNK_001: Conectividad Splunk
- CONFIG_001: Configuración inválida
- DEPLOY_001: Deployment falla
- **Fix**: `kepler setup <platform>` + `kepler validate`

---

> **💡 Tip:** Siempre ejecuta `kepler validate ecosystem` antes de empezar un proyecto nuevo. Te ahorrará tiempo identificando problemas temprano.

> **🎯 Estado Actual:** Sistema de validación completo implementado con mensajes accionables, auto-fixes, y troubleshooting inteligente.
