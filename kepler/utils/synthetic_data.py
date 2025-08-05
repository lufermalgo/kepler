"""
Generador de datos sintéticos para simulación de métricas industriales.

Este módulo genera datos realistas para laboratorio y testing del framework Kepler.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
from pathlib import Path


class IndustrialDataGenerator:
    """
    Generador de datos sintéticos para simular métricas industriales.
    """
    
    def __init__(self, seed: Optional[int] = 42):
        """
        Inicializa el generador con una semilla para reproducibilidad.
        
        Args:
            seed: Semilla para random y numpy
        """
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_sensor_metrics(
        self, 
        duration_hours: int = 24,
        interval_minutes: int = 5,
        num_sensors: int = 10
    ) -> pd.DataFrame:
        """
        Genera métricas de sensores industriales.
        
        Args:
            duration_hours: Duración de la simulación en horas
            interval_minutes: Intervalo entre lecturas en minutos
            num_sensors: Número de sensores a simular
            
        Returns:
            DataFrame con métricas de sensores
        """
        # Generar timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        
        timestamps = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{interval_minutes}min'
        )
        
        data = []
        
        # Tipos de sensores industriales
        sensor_types = {
            'temperature': {'min': 20, 'max': 85, 'unit': '°C', 'noise': 2.0},
            'pressure': {'min': 1.0, 'max': 15.0, 'unit': 'bar', 'noise': 0.3},
            'flow_rate': {'min': 0, 'max': 500, 'unit': 'L/min', 'noise': 10.0},
            'vibration': {'min': 0.1, 'max': 5.0, 'unit': 'mm/s', 'noise': 0.2},
            'power_consumption': {'min': 50, 'max': 1500, 'unit': 'kW', 'noise': 25.0}
        }
        
        # Generar datos para cada sensor
        for sensor_id in range(1, num_sensors + 1):
            sensor_type = random.choice(list(sensor_types.keys()))
            config = sensor_types[sensor_type]
            
            # Área de la planta donde está el sensor
            area = random.choice(['production', 'utilities', 'storage', 'quality', 'maintenance'])
            
            # Generar serie temporal con tendencias y patrones
            base_values = self._generate_time_series(
                timestamps, 
                config['min'], 
                config['max'],
                config['noise']
            )
            
            for i, (timestamp, value) in enumerate(zip(timestamps, base_values)):
                # Simular ocasionalmente valores anómalos
                if random.random() < 0.02:  # 2% de probabilidad de anomalía
                    value = self._generate_anomaly(value, config)
                
                # Simular ocasionalmente datos faltantes
                if random.random() < 0.01:  # 1% de probabilidad de dato faltante
                    value = None
                
                data.append({
                    'timestamp': timestamp,
                    'sensor_id': f'SENSOR_{sensor_id:03d}',
                    'sensor_type': sensor_type,
                    'area': area,
                    'value': value,
                    'unit': config['unit'],
                    'status': 'normal' if value is not None else 'offline',
                    'facility': 'Plant_Alpha'
                })
        
        df = pd.DataFrame(data)
        
        # Agregar algunas métricas calculadas
        df = self._add_calculated_metrics(df)
        
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def generate_production_metrics(
        self,
        duration_hours: int = 24,
        interval_minutes: int = 15
    ) -> pd.DataFrame:
        """
        Genera métricas de producción industrial.
        
        Args:
            duration_hours: Duración de la simulación en horas
            interval_minutes: Intervalo entre mediciones
            
        Returns:
            DataFrame con métricas de producción
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        
        timestamps = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{interval_minutes}min'
        )
        
        data = []
        
        # Líneas de producción
        production_lines = ['Line_A', 'Line_B', 'Line_C']
        
        for timestamp in timestamps:
            for line in production_lines:
                # Simular patrones de producción (turno diurno más activo)
                hour = timestamp.hour
                if 6 <= hour <= 18:  # Turno diurno
                    base_efficiency = 0.85
                    base_throughput = 150
                elif 18 <= hour <= 22:  # Turno vespertino
                    base_efficiency = 0.75
                    base_throughput = 120
                else:  # Turno nocturno
                    base_efficiency = 0.65
                    base_throughput = 80
                
                # Agregar variabilidad
                efficiency = max(0, min(1, base_efficiency + np.random.normal(0, 0.1)))
                throughput = max(0, base_throughput + np.random.normal(0, 15))
                
                # Calcular métricas derivadas
                target_throughput = 180
                oee = efficiency * (throughput / target_throughput) * 0.95  # 95% availability
                
                # Simular downtime ocasional
                is_downtime = random.random() < 0.05  # 5% probabilidad
                if is_downtime:
                    efficiency = 0
                    throughput = 0
                    oee = 0
                
                data.append({
                    'timestamp': timestamp,
                    'production_line': line,
                    'efficiency': efficiency,
                    'throughput': throughput,
                    'target_throughput': target_throughput,
                    'oee': oee,
                    'downtime': is_downtime,
                    'shift': self._get_shift(hour),
                    'facility': 'Plant_Alpha'
                })
        
        return pd.DataFrame(data)
    
    def generate_quality_metrics(
        self,
        duration_hours: int = 24,
        interval_minutes: int = 30
    ) -> pd.DataFrame:
        """
        Genera métricas de calidad industrial.
        
        Args:
            duration_hours: Duración de la simulación
            interval_minutes: Intervalo entre mediciones
            
        Returns:
            DataFrame con métricas de calidad
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=duration_hours)
        
        timestamps = pd.date_range(
            start=start_time,
            end=end_time,
            freq=f'{interval_minutes}min'
        )
        
        data = []
        
        quality_tests = [
            {'name': 'dimensional_tolerance', 'target': 0.5, 'tolerance': 0.1, 'unit': 'mm'},
            {'name': 'surface_roughness', 'target': 1.6, 'tolerance': 0.3, 'unit': 'Ra'},
            {'name': 'hardness', 'target': 45, 'tolerance': 3, 'unit': 'HRC'},
            {'name': 'chemical_composition', 'target': 98.5, 'tolerance': 1.0, 'unit': '%'}
        ]
        
        for timestamp in timestamps:
            for test in quality_tests:
                # Generar valor con distribución normal centrada en target
                value = np.random.normal(test['target'], test['tolerance'] / 3)
                
                # Determinar si pasa o falla el test
                lower_limit = test['target'] - test['tolerance']
                upper_limit = test['target'] + test['tolerance']
                passed = lower_limit <= value <= upper_limit
                
                # Simular ocasionalmente fallos críticos
                if random.random() < 0.03:  # 3% de fallos críticos
                    value = test['target'] + random.choice([-1, 1]) * test['tolerance'] * 2
                    passed = False
                
                data.append({
                    'timestamp': timestamp,
                    'test_name': test['name'],
                    'measured_value': value,
                    'target_value': test['target'],
                    'lower_limit': lower_limit,
                    'upper_limit': upper_limit,
                    'unit': test['unit'],
                    'passed': passed,
                    'batch_id': f"BATCH_{timestamp.strftime('%Y%m%d_%H%M')}",
                    'facility': 'Plant_Alpha'
                })
        
        return pd.DataFrame(data)
    
    def _generate_time_series(
        self, 
        timestamps: pd.DatetimeIndex,
        min_val: float,
        max_val: float,
        noise_std: float
    ) -> np.ndarray:
        """
        Genera una serie temporal con tendencias y patrones cíclicos.
        
        Args:
            timestamps: Timestamps para la serie
            min_val: Valor mínimo
            max_val: Valor máximo
            noise_std: Desviación estándar del ruido
            
        Returns:
            Array con valores de la serie temporal
        """
        n = len(timestamps)
        
        # Tendencia base
        base_trend = np.linspace(min_val, max_val, n) * 0.7 + (max_val + min_val) / 2 * 0.3
        
        # Patrón cíclico diario (24 horas)
        daily_cycle = np.sin(2 * np.pi * np.arange(n) / (24 * 60 / 5)) * (max_val - min_val) * 0.2
        
        # Patrón cíclico semanal
        weekly_cycle = np.cos(2 * np.pi * np.arange(n) / (7 * 24 * 60 / 5)) * (max_val - min_val) * 0.1
        
        # Ruido aleatorio
        noise = np.random.normal(0, noise_std, n)
        
        # Combinar componentes
        values = base_trend + daily_cycle + weekly_cycle + noise
        
        # Asegurar que esté dentro de los límites
        values = np.clip(values, min_val, max_val)
        
        return values
    
    def _generate_anomaly(self, normal_value: float, config: Dict) -> float:
        """
        Genera un valor anómalo basado en el valor normal.
        
        Args:
            normal_value: Valor normal
            config: Configuración del sensor
            
        Returns:
            Valor anómalo
        """
        # Tipo de anomalía
        anomaly_type = random.choice(['spike', 'drop', 'drift'])
        
        if anomaly_type == 'spike':
            # Pico hacia arriba
            return min(config['max'] * 1.2, normal_value * random.uniform(1.5, 3.0))
        elif anomaly_type == 'drop':
            # Caída hacia abajo
            return max(config['min'] * 0.8, normal_value * random.uniform(0.1, 0.5))
        else:  # drift
            # Deriva gradual
            return normal_value * random.uniform(0.7, 1.3)
    
    def _add_calculated_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega métricas calculadas al DataFrame.
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame con métricas adicionales
        """
        # Agregar columnas de tiempo
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['shift'] = df['hour'].apply(self._get_shift)
        
        # Agregar indicadores de alerta
        df['alert_level'] = 'normal'
        
        # Lógica simple de alertas por tipo de sensor
        for sensor_type in df['sensor_type'].unique():
            mask = df['sensor_type'] == sensor_type
            if sensor_type == 'temperature':
                df.loc[mask & (df['value'] > 80), 'alert_level'] = 'high'
                df.loc[mask & (df['value'] > 90), 'alert_level'] = 'critical'
            elif sensor_type == 'pressure':
                df.loc[mask & (df['value'] > 12), 'alert_level'] = 'high'
                df.loc[mask & (df['value'] > 14), 'alert_level'] = 'critical'
            elif sensor_type == 'vibration':
                df.loc[mask & (df['value'] > 3), 'alert_level'] = 'high'
                df.loc[mask & (df['value'] > 4), 'alert_level'] = 'critical'
        
        return df
    
    def _get_shift(self, hour: int) -> str:
        """
        Determina el turno basado en la hora.
        
        Args:
            hour: Hora del día (0-23)
            
        Returns:
            Nombre del turno
        """
        if 6 <= hour < 14:
            return 'morning'
        elif 14 <= hour < 22:
            return 'afternoon'
        else:
            return 'night'
    
    def export_to_splunk_events(self, df: pd.DataFrame, output_file: Optional[str] = None) -> List[Dict]:
        """
        Convierte DataFrame a formato de eventos para Splunk.
        
        Args:
            df: DataFrame con datos
            output_file: Archivo opcional para guardar eventos
            
        Returns:
            Lista de eventos para Splunk
        """
        events = []
        
        for _, row in df.iterrows():
            # Convertir el row a dict y manejar timestamps
            row_dict = row.to_dict()
            
            # Convertir timestamps a strings para serialización JSON
            for key, value in row_dict.items():
                if pd.isna(value):
                    row_dict[key] = None
                elif hasattr(value, 'isoformat'):  # datetime objects
                    row_dict[key] = value.isoformat()
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    row_dict[key] = pd.to_datetime(value).isoformat()
            
            event = {
                'time': int(row['timestamp'].timestamp()),
                'source': 'kepler_simulator',
                'sourcetype': 'industrial_metrics',
                'index': 'kepler_lab',
                'event': row_dict
            }
            events.append(event)
        
        if output_file:
            import json
            with open(output_file, 'w') as f:
                for event in events:
                    f.write(json.dumps(event) + '\n')
        
        return events
    
    def export_to_splunk_metrics(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convierte DataFrame a formato de métricas para Splunk según documentación oficial.
        
        FORMATO CORRECTO CONFIRMADO por pruebas:
        - Una métrica por evento
        - 'metric_name': '<nombre>' (sin prefijo 'metric_name:')
        - '_value': <número>
        - Búsquedas: usar 'mstats', no 'search'
        
        Args:
            df: DataFrame con datos
            
        Returns:
            Lista de métricas individuales para Splunk
        """
        metrics = []
        
        # Generar UNA métrica por fila (formato correcto)
        for _, row in df.iterrows():
            if pd.notna(row.get('value')):
                timestamp = row['timestamp']
                
                # Crear métrica individual (formato correcto confirmado)
                metric_data = {
                    'time': int(timestamp.timestamp()),
                    'event': 'metric',  # OBLIGATORIO para métricas
                    'source': 'kepler_simulator',
                    'sourcetype': 'industrial_metrics', 
                    'index': 'kepler_metrics',
                    'host': 'kepler-lab',
                    'fields': {
                        # FORMATO CORRECTO: campo separado + _value
                        'metric_name': f"{row.get('sensor_type', 'metric')}.{row.get('sensor_id', 'unknown')}",
                        '_value': float(row['value']),
                        
                        # Dimensiones (metadatos)
                        'sensor_id': row.get('sensor_id', 'unknown'),
                        'sensor_type': row.get('sensor_type', 'metric'),
                        'unit': row.get('unit', ''),
                        'status': row.get('status', 'normal'),
                        'facility': row.get('facility', ''),
                        'area': row.get('area', '')
                    }
                }
                
                # Limpiar campos vacíos
                metric_data['fields'] = {k: v for k, v in metric_data['fields'].items() 
                                       if v is not None and v != ''}
                
                metrics.append(metric_data)
        
        return metrics


def create_lab_dataset(duration_hours: int = 48) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Crea un dataset completo para laboratorio.
    
    Args:
        duration_hours: Duración del dataset en horas
        
    Returns:
        Tuple con (sensor_data, production_data, quality_data)
    """
    generator = IndustrialDataGenerator()
    
    # Generar diferentes tipos de datos
    sensor_data = generator.generate_sensor_metrics(duration_hours=duration_hours)
    production_data = generator.generate_production_metrics(duration_hours=duration_hours)
    quality_data = generator.generate_quality_metrics(duration_hours=duration_hours)
    
    return sensor_data, production_data, quality_data


if __name__ == "__main__":
    # Ejemplo de uso
    generator = IndustrialDataGenerator()
    
    # Generar datos de sensores
    sensor_df = generator.generate_sensor_metrics(duration_hours=24, num_sensors=5)
    print(f"Generated {len(sensor_df)} sensor readings")
    print(sensor_df.head())
    
    # Generar datos de producción
    production_df = generator.generate_production_metrics(duration_hours=24)
    print(f"\nGenerated {len(production_df)} production records")
    print(production_df.head())
    
    # Exportar para Splunk
    events = generator.export_to_splunk_events(sensor_df, 'sensor_events.json')
    metrics = generator.export_to_splunk_metrics(sensor_df)
    
    print(f"\nExported {len(events)} events and {len(metrics)} metrics for Splunk")