# postprocess — README

## Descripción

Este repositorio contiene el script **`postprocess.py`** que procesa datos de llegadas de autobuses (eventos de paradas).

**Funcionalidades principales:**
- Extrae eventos de paradas de autobuses desde archivos CSV de datos brutos
- Detecta viajes consecutivos de autobuses (reconocimiento automático de cambio de viaje)
- Calcula el **headway** (intervalo entre buses) por parada y sentido
- Identifica paradas faltantes en cada viaje
- Genera reportes con estadísticas de headway (media, mediana, desviación estándar)
- Exporta eventos procesados a CSV con información detallada de cada paso

## Requisitos

- **Python 3.8+**
- Dependencias listadas en `requirements.txt`:
  - pandas
  - numpy
  - matplotlib
  - seaborn

## Instalación rápida

1. Navega al directorio del proyecto:
```bash
cd /Directory/
```

2. Crea un entorno virtual (opcional pero recomendado):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Configuración previa

Antes de ejecutar el script, configura las variables globales en `postprocess.py` (líneas 14-20):

```python
PARADES_FILE       = "parades.json"        # Archivo JSON con información de paradas (opcional)
MAX_ORDRE_JUMP     = 5                     # Diferencia máxima en orden de parada entre registros
MIN_DIFF_VIAJE     = 45                    # Minutos mínimos para considerar un nuevo viaje
RESTART_JUMP       = 10                    # Salto mínimo de orden para detectar reinicio de viaje
MAX_ETA_MINUTES    = 60                    # ETA máximo permitido (minutos desde captura)
INPUT_PATH         = "data/arrivals_2026-05-02_8_2.csv"  # Archivo o carpeta de entrada
OUTPUT_EVENTOS_DIR = "eventos_nuevos/"      # Carpeta de salida
```

### Parámetros importantes

| Parámetro | Descripción |
|-----------|-------------|
| `INPUT_PATH` | Ruta al archivo CSV o carpeta con archivos CSV para procesar |
| `OUTPUT_EVENTOS_DIR` | Carpeta destino para los resultados procesados |
| `PARADES_FILE` | JSON opcional con orden de paradas (mapa sentido → parada → orden) |
| `MIN_DIFF_VIAJE` | Umbral temporal para detectar cambio de viaje (en minutos) |
| `MAX_ETA_MINUTES` | Rango máximo de ETA a considerar válido |

## Uso básico

### Opción 1: Procesar un archivo único

1. Actualiza `INPUT_PATH` en `postprocess.py`:
```python
INPUT_PATH = "data/arrivals_2026-05-02_8_2.csv"
```

2. Ejecuta el script:
```bash
python postprocess.py
```

### Opción 2: Procesar todos los archivos de una carpeta

1. Actualiza `INPUT_PATH` para que apunte a un directorio:
```python
INPUT_PATH = "data/"
```

2. Ejecuta el script:
```bash
python postprocess.py
```

El script procesará automáticamente todos los archivos `*.csv` de la carpeta que no contengan "limpio" en el nombre.

## Salidas generadas

El script genera los siguientes archivos en la carpeta especificada en `OUTPUT_EVENTOS_DIR`:

| Archivo | Descripción |
|---------|-------------|
| `{archivo}_limpio_3.csv` | Eventos procesados con timestamp de paso, headway y paradas faltantes |
| `{archivo}_headway_medio.csv` | Estadísticas de headway agregadas por parada y sentido |

### Columnas en el archivo de salida

- `id_bus`: Identificador del autobús
- `viaje_n`: Número de viaje del autobús
- `nom_linia`: Nombre de la línea
- `sentit`: Sentido (Anada/Tornada)
- `ordre`: Orden de la parada en el itinerario
- `nom_parada`: Nombre de la parada
- `codi_parada`: Código de la parada
- `hora_paso`: Hora de paso (HH:MM:SS)
- `min_paso`: Minuto de paso con decimales
- `headway_min`: Intervalo entre buses en minutos
- `headway_sec`: Intervalo entre buses en segundos
- `ordre_faltantes`: Órdenes de paradas no registradas en el viaje

## Monitoreo y resultados

Durante la ejecución, el script imprime:

1. **Información de procesamiento**: Logs de cada archivo procesado
2. **Análisis de paradas faltantes**: Por bus y viaje
3. **Estadísticas de headway**:
   - Headway medio global
   - Tabla detallada por parada (media, mediana, desviación estándar, count)
   - Desviación estándar media
4. **Gráficos visuales**: Histogramas y boxplots de distribución de headways

## Solución de problemas

| Problema | Solución |
|----------|----------|
| **FileNotFoundError** | Verifica que `INPUT_PATH` sea correcto y el archivo/carpeta exista |
| **KeyError en columnas** | Asegúrate de que el CSV contiene: `id_bus`, `sentit`, `codi_parada`, `temps_arribada`, `captured_at`, `ordre`, `nom_parada`, `nom_linia` |
| **No se detectan eventos** | Aumenta `MAX_ETA_MINUTES` o revisa los valores de `temps_arribada` (deben ser > 0 y < 60 min) |
| **Headway muy bajo/alto** | El script filtra automáticamente con IQR para eliminar outliers (rango: Q1-1.5×IQR a Q3+1.5×IQR) |
| **Paradas faltantes elevadas** | Verifica que `MAX_ORDRE_JUMP` no sea muy restrictivo; aumenta si hay saltos normales en el orden |

## Ejemplo de ejecución completa

```bash
# 1. Activar entorno (si existe)
source .venv/bin/activate

# 2. Editar postprocess.py para configurar INPUT_PATH y OUTPUT_EVENTOS_DIR
# (editar líneas 14-20)

# 3. Ejecutar
python postprocess.py

# 4. Revisar resultados en la carpeta eventos_nuevos/
ls -la eventos_nuevos/
```

## Dependencias del proyecto

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

Para actualizar dependencias, edita `requirements.txt` y ejecuta:
```bash
pip install -r requirements.txt --upgrade
```
