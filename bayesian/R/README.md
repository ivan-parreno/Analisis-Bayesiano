# Modelo 1 — Reloj Estático (R + Stan)

Estimación bayesiana del headway de referencia por **parada**, **franja horaria** y **día de la semana** para la línea H8 · Anada.

## Modelo

$$\log(\mu_i) = \alpha + a_{s[i]} + b_{h[i]} + c_{d[i]}$$

| Componente | Prior |
|------------|-------|
| $\alpha$ | $\mathcal{N}(2.0,\, 0.4)$ |
| $a_s$ | $\mathcal{N}(0,\, \sigma_s)$, $\sigma_s \sim \mathcal{N}^+(0.3)$ |
| $b_h$ | Gaussian Random Walk, paso $\sigma_t \sim \mathcal{N}^+(0.2)$ |
| $c_d$ | $\mathcal{N}(0,\, \sigma_d)$, $\sigma_d \sim \mathcal{N}^+(0.15)$ |
| Dispersión | $\kappa \sim \mathrm{Exp}(0.5)$ global |

Likelihood (Gamma desplazada, soporte $y > 0.16$ min):

$$y_{\text{shifted}} = y_{\text{obs}} - 0.16 \sim \mathrm{Gamma}\!\left(\kappa,\; \frac{\kappa}{\mu}\right)$$

## Requisitos

- R ≥ 4.2
- Paquetes: `rstan`, `posterior`, `readr`, `jsonlite`, `ggplot2`

```r
install.packages(c("rstan", "posterior", "readr", "jsonlite", "ggplot2"))
```

`rstan` requiere una toolchain C++ (Xcode CLT en macOS).

## Flujo

### Opción 0 — Un solo script en RStudio (más fácil)

Abre `run_model1.R` → **Ctrl+A** → **Enter** (o botón **Source**).

Edita arriba del archivo:
- **Un día:** `FECHA <- "2026-06-06"`
- **Rango:** `FECHA <- NULL`, `FECHA_INICIO <- "2026-05-01"`, `FECHA_FIN <- "2026-05-31"`
- **Todo:** los tres en `NULL`
- `ITER`, `WARMUP`, `CHAINS` según necesites

### Opción A — Directo desde `processed/headways/` (recomendado)

Lee los CSV `headways_pares3_H8_Anada_*.csv` sin pasar por Python.

```bash
cd bayesian/R

# Un día
Rscript fit_model1.R --headways ../../processed/headways --date 2026-06-06 --chains 2 --iter 400 --warmup 200

# Rango de fechas (inclusive)
Rscript fit_model1.R --headways ../../processed/headways --from 2026-05-01 --to 2026-05-31

# Histórico completo
Rscript fit_model1.R --headways ../../processed/headways --chains 4 --iter 1600 --warmup 800
```

En RStudio:

```r
setwd("/Users/ivan/Downloads/AWS/data/bayesian/R")
source("load_pares.R")
loaded <- load_stan_data_from_pares("../../processed/headways", date = "2026-06-06")
loaded$stan_data   # listo para stan()
```

### Opción B — CSV intermedio (Python)

```bash
cd bayesian
python prepare_bayes_data.py --source pairs --date 2026-06-06 --stop-only
cd R
Rscript fit_model1.R --data ../data/2026-06-06
```

Salida en `R/output/<nombre_data>/`:

| Archivo | Contenido |
|---------|-----------|
| `fit_model1.rds` | Objeto `stanfit` |
| `summary_model1.csv` | Resumen de parámetros |
| `mu_ref.csv` / `mu_ref.rds` | Matriz $\mu_{\text{ref}}[s,h,d]$ (media posterior) |

### 3. Diagnósticos

```bash
Rscript plot_model1.R --fit output/2026-06-06/fit_model1.rds
```

## Índices

Los CSV usan índices **0-based** (`s`, `h`, `d`). Stan recibe índices **1-based** (conversión automática en `fit_model1.R`).

| `h` | Franja |
|-----|--------|
| 0 | Madrugada (00–06:59) |
| 1–15 | 07:00–21:59 (una hora cada una) |
| 16 | Noche (22:00–23:59) |

| `d` | Día |
|-----|-----|
| 0–6 | Lunes – Domingo |

## Notas

- Fuente recomendada: **pares** (`--source pairs`), headway delante→detrás por parada.
- Filtros: headway 1–30 min, `y_shifted > 0`.
- Modelo 2 (propagación dinámica en pares) se implementará en Stan en una fase posterior.
