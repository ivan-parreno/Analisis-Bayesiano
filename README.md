# Pipeline de limpieza — datos de autobuses H8

## Estructura del repositorio

```
data/
├── raw/                    # Datos brutos de la API TMB
│   ├── data15/             # Mayo 3–15
│   └── data16/             # Mayo 16 – Junio 6
├── processed/
│   ├── eventos/            # Llegadas limpias (*_limpio_3.csv)
│   └── headways/           # Headways entre pares de viajes
├── bayesian/               # Modelo bayesiano (R + Stan)
│   ├── prepare_bayes_data.py
│   └── R/                  # Stan + scripts R
├── pipeline.py             # Orquestador principal
├── postprocess.py          # Brutos → llegadas
├── plot_hours.py           # Llegadas → headways de pares
├── script.py               # Captura en vivo vía API TMB
└── parades.json            # Orden de paradas (opcional)
```

## Productos finales

Por cada día procesado se generan **dos CSVs obligatorios**:

| Producto | Carpeta | Ejemplo |
|----------|---------|---------|
| Llegadas | `processed/eventos/` | `arrivals_2026-05-16_8_2_limpio_3.csv` |
| Headways pares | `processed/headways/` | `headways_pares3_H8_Anada_2026-05-16.csv` |

## Requisitos

- Python 3.11+
- Dependencias en `pyproject.toml` (o `requirements.txt`)

```bash
cd /Users/ivan/Downloads/AWS/data
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# o: uv sync
```

## Pipeline de limpieza

### Procesar datos nuevos (brutos → llegadas + headways)

```bash
python pipeline.py --input raw/data16
```

### Solo generar headways de pares (desde limpios ya existentes)

```bash
python pipeline.py --headways-only --eventos processed/eventos
```

### Un día concreto

```bash
python pipeline.py --input raw/data16 --date 2026-05-16
```

### Scripts individuales

```bash
# Solo llegadas
python postprocess.py --input raw/data16 --output processed/eventos --no-figures

# Solo headways de pares (sin gráficos)
python plot_hours.py --csv processed/eventos/arrivals_2026-05-16_8_2_limpio_3.csv --export-only
```

## Visualizar headways de pares (HTML)

Genera stringlines interactivos (como antes) en `processed/figures/`:

```bash
# Un día
python visualize_pares.py --date 2026-06-06

# Todos los días + índice navegable
python visualize_pares.py --all
```

Abre **`processed/figures/visualize.html`** en el navegador: selector de fecha + stringline de buses en pares + gráfico de headway por par (dropdown).

## Análisis bayesiano (R + Stan)

Modelo 1 (Reloj Estático) según `bayesian/PLAN_MODELO.md`. Detalle en `bayesian/R/README.md`.

```bash
cd bayesian

# Preparar datos (pares delante→detrás, recomendado)
python prepare_bayes_data.py --source pairs --stop-only

# Prueba con un día
python prepare_bayes_data.py --date 2026-06-06 --stop-only

# Ajustar en R
cd R
Rscript fit_model1.R --data ../data/2026-06-06 --iter 400 --warmup 200
Rscript plot_model1.R --fit output/2026-06-06/fit_model1.rds
```

## Parámetros de postprocess

| Parámetro | Descripción |
|-----------|-------------|
| `--input` | Carpeta o archivo CSV bruto |
| `--output` | Carpeta destino para `*_limpio_3.csv` |
| `--date` | Filtrar por fecha `YYYY-MM-DD` |
| `--no-figures` | No mostrar gráficos |

## Columnas en llegadas (`*_limpio_3.csv`)

- `id_bus`, `viaje_n`, `nom_linia`, `sentit`, `ordre`, `nom_parada`, `codi_parada`
- `hora_paso`, `min_paso`, `headway_min`, `headway_sec`, `ordre_faltantes`

## Columnas en headways de pares

- `par_id`, `bus_delante`, `trip_delante`, `bus_detras`, `trip_detras`
- `codi_parada`, `ordre`, `headway_pair`
- `hora_delante_str`, `hora_detras_str`, `franja_hora`
