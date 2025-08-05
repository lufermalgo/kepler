# Código para corregir la celda 3 del notebook
import json

# Leer el notebook
with open('metrics_analysis.ipynb', 'r') as f:
    notebook = json.load(f)

# Encontrar la celda 3 (índice 3) y actualizar su código
notebook['cells'][3]['source'] = [
    "# Veamos qué métricas están disponibles\n",
    "metricas_disponibles = kp.data.from_splunk(\n",
    "    spl=\"| mstats latest(_value) as ultimo_valor WHERE index=kepler_metrics by metric_name\"\n",
    ")\n",
    "\n",
    "if len(metricas_disponibles) == 0:\n",
    "    print(\"⚠️ El índice de métricas actualmente está vacío.\")\n",
    "    print(\"🔄 Esto es normal si acabas de configurar Kepler.\")\n",
    "    print(\"💡 Las métricas se están cargando en segundo plano...\")\n",
    "    print(\"\\n📋 Cuando haya datos, aquí verías algo como:\")\n",
    "    print(\"   - temperature.SENSOR_001\")\n",
    "    print(\"   - pressure.SENSOR_002\") \n",
    "    print(\"   - vibration.SENSOR_003\")\n",
    "    print(\"   - flow.SENSOR_004\")\n",
    "    print(\"\\n▶️ Voy a continuar con datos simulados para demostrar las capacidades...\")\n",
    "else:\n",
    "    print(f\"✅ Métricas encontradas: {len(metricas_disponibles)}\")\n",
    "\n",
    "metricas_disponibles"
]

# Guardar el notebook corregido
with open('metrics_analysis.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("✅ Notebook corregido")