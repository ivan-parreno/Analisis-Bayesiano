# =============================================================================
# Modelo de Inestabilidad Irrecuperable — headways en MINUTOS
# ¿Qué parada / hora / convoy genera variabilidad que no se recupera aguas abajo?
# Usa trenes de buses encadenados cuando existen; pares sueltos se modelan aparte.
# =============================================================================

FECHA_INICIO <- "2026-05-02"
FECHA_FIN    <- "2026-06-06"

CHAINS  <- 2
ITER    <- 2000
WARMUP  <- 500
CORES   <- 2
SEED    <- 42

BASE_REPO    <- "/Users/ivan/Downloads/AWS/data"
HEADWAYS_DIR <- file.path(BASE_REPO, "processed", "headways")
STAN_FILE    <- file.path(BASE_REPO, "bayesian", "R", "stan", "instability_minutes.stan")
OUT_DIR      <- file.path(BASE_REPO, "bayesian", "R", "output_instability_2")

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

library(data.table)
library(ggplot2)
library(rstan)
library(lubridate)
library(splines)

# --- Asignar trenes de buses (platoons) por fecha ---
# Encadenamiento: trip_detras del par A = trip_delante del par B
assign_platoons <- function(dt) {
  pairs <- unique(dt[, .(par_id, trip_delante, trip_detras, fecha)])
  out <- pairs[, .(par_id, fecha, platoon_key = paste0(fecha, "__solo"))]
  out[, platoon_idx := 0L]

  for (f in unique(pairs$fecha)) {
    day <- pairs[fecha == f]
    if (nrow(day) < 2) next

    par_ids <- day$par_id
    parent <- setNames(par_ids, par_ids)

    find_root <- function(x) {
      while (!identical(parent[[x]], x)) {
        parent[[x]] <<- parent[[parent[[x]]]]
        x <- parent[[x]]
      }
      x
    }
    unite <- function(a, b) {
      ra <- find_root(a)
      rb <- find_root(b)
      if (!identical(ra, rb)) parent[[rb]] <<- ra
    }

    edges <- merge(
      day[, .(from = par_id, mid = trip_detras)],
      day[, .(to = par_id, mid = trip_delante)],
      by = "mid"
    )
    edges <- edges[from != to]
    if (nrow(edges) == 0) next

    for (i in seq_len(nrow(edges))) {
      unite(edges$from[i], edges$to[i])
    }

    roots <- vapply(par_ids, find_root, character(1))
    tab <- table(roots)
    multi_roots <- names(tab[tab > 1])
    if (length(multi_roots) == 0) next

    for (r in multi_roots) {
      members <- par_ids[roots == r]
      key <- paste0(f, "__", r)
      out[fecha == f & par_id %in% members, platoon_key := key]
    }
  }

  keys <- unique(out[!endsWith(platoon_key, "__solo"), platoon_key])
  if (length(keys) > 0) {
    key_idx <- setNames(seq_along(keys), keys)
    out[platoon_key %in% keys, platoon_idx := as.integer(key_idx[platoon_key])]
  }
  out[, .(par_id, fecha, platoon_idx, platoon_key)]
}

load_pares_raw <- function(directory, from, to) {
  files <- list.files(
    directory,
    pattern = "^headways_pares3_H8_Anada_\\d{4}-\\d{2}-\\d{2}\\.csv$",
    full.names = TRUE
  )
  files <- files[file.exists(files) & file.info(files)$size > 0]
  if (length(files) == 0) stop("No se encontraron archivos válidos en: ", directory)

  fechas <- as.Date(sub(".*_(\\d{4}-\\d{2}-\\d{2})\\.csv$", "\\1", basename(files)))
  files  <- files[fechas >= as.Date(from) & fechas <= as.Date(to)]
  if (length(files) == 0) stop("No hay archivos en el rango de fechas seleccionado")

  dt_list <- lapply(files, function(f) {
    dt <- tryCatch(
      fread(f, colClasses = list(character = c("hora_delante_str", "hora_detras_str"))),
      error = function(e) { message("Error leyendo: ", f); NULL }
    )
    if (!is.null(dt)) {
      dt[, fecha := as.Date(sub(".*_(\\d{4}-\\d{2}-\\d{2})\\.csv$", "\\1", basename(f)))]
      if ("headway_pair" %in% names(dt)) setnames(dt, "headway_pair", "headway")
    }
    dt
  })
  rbindlist(dt_list[!sapply(dt_list, is.null)], fill = TRUE)
}

build_trip_index <- function(dt, trip_ids) {
  trips_local <- unique(dt[trip_id %in% trip_ids], by = "trip_id")
  setorder(trips_local, par_id, fecha)
  P_local <- nrow(trips_local)
  N_local <- integer(P_local)
  pos_local <- integer(P_local + 1)
  pos_local[1] <- 1L
  hora_local <- numeric(P_local)
  platoon_local <- integer(P_local)

  for (i in seq_len(P_local)) {
    sub <- dt[trip_id == trips_local$trip_id[i]]
    n <- nrow(sub)
    N_local[i] <- n
    pos_local[i + 1] <- pos_local[i] + n
    h1 <- sub[ordre == 1, hora_dec]
    hora_local[i] <- if (length(h1) >= 1 && !is.na(h1[1])) h1[1] else sub[1, hora_dec]
    platoon_local[i] <- sub[1, platoon_idx]
  }
  list(
    P = P_local, trips = trips_local, N_stops = N_local,
    pos = pos_local, hora_trip = hora_local, platoon_trip = platoon_local
  )
}

# --- 1. CARGA ---
raw <- load_pares_raw(HEADWAYS_DIR, from = FECHA_INICIO, to = FECHA_FIN)
if (!"headway" %in% names(raw)) stop("Falta columna 'headway'")
raw <- raw[headway >= 1 & headway <= 30]

raw[, weekend := fifelse(wday(fecha) %in% c(1, 7), 1, 0)]
raw[, hora_time := as.POSIXct(hora_delante_str, format = "%H:%M:%S", tz = "UTC")]
raw[, hora_dec := hour(hora_time) + minute(hora_time) / 60 + second(hora_time) / 3600]

# Platoon por par_id × fecha
plat_map <- assign_platoons(raw)
raw <- merge(raw, plat_map, by = c("par_id", "fecha"), all.x = TRUE)
raw[is.na(platoon_idx), platoon_idx := 0L]

setorder(raw, par_id, fecha, ordre)
raw[, trip_id := paste(par_id, fecha, sep = "__")]

min_ordre <- raw[, min(ordre), by = trip_id]
raw <- raw[trip_id %in% min_ordre[V1 == 1, trip_id]]
if (nrow(raw) == 0) stop("No hay trayectos con parada 1 observada")

# --- 2. ÍNDICES STAN ---
trips <- unique(raw, by = "trip_id")
setorder(trips, par_id, fecha)
P <- nrow(trips)
S_real <- max(raw$ordre)

idx <- build_trip_index(raw, raw$trip_id)
P <- idx$P
trips <- idx$trips
N_stops <- idx$N_stops
pos <- idx$pos
hora_trip <- idx$hora_trip
platoon_trip <- idx$platoon_trip
N_platoon <- max(platoon_trip)
if (N_platoon == 0L) N_platoon <- 1L  # dummy Stan; platoon_idx=0 en todos

if (any(N_stops > S_real)) stop("N_stops > S en algún trayecto")
if (any(is.na(hora_trip))) stop("Hay trayectos sin hora en parada 1")

hora_ok <- hora_trip >= 6 & hora_trip <= 22
if (!all(hora_ok)) {
  message("Recortando ", sum(!hora_ok), " trayectos fuera de 6-22 h")
  raw <- raw[trip_id %in% trips$trip_id[hora_ok]]
  setorder(raw, par_id, fecha, ordre)
  idx <- build_trip_index(raw, raw$trip_id)
  P <- idx$P
  trips <- idx$trips
  N_stops <- idx$N_stops
  pos <- idx$pos
  hora_trip <- idx$hora_trip
  platoon_trip <- idx$platoon_trip
  N_platoon <- max(platoon_trip)
  if (N_platoon == 0L) N_platoon <- 1L
}

B_trip <- bs(hora_trip, df = 5, degree = 3)

n_en_tren <- sum(platoon_trip > 0)
n_sueltos <- sum(platoon_trip == 0)
cat(sprintf(
  paste0(
    "Trayectos: %d | Obs: %d | En convoy: %d | Sueltos: %d | ",
    "Platoons: %d | Paradas max: %d\n"
  ),
  P, nrow(raw), n_en_tren, n_sueltos, N_platoon, S_real
))

stan_data <- list(
  P = P,
  S = S_real,
  N_stops = N_stops,
  N_total = nrow(raw),
  stop = raw$ordre,
  y = raw$headway,
  weekend = raw$weekend,
  pos = pos,
  hora_trip = hora_trip,
  K = ncol(B_trip),
  B_trip = B_trip,
  N_platoon = N_platoon,
  platoon_idx = platoon_trip
)

# --- 3. MUESTREO ---
rstan_options(auto_write = TRUE)
options(mc.cores = CORES)

cat("Ajustando modelo de inestabilidad (minutos)...\n")
t0 <- Sys.time()
fit <- stan(
  file = STAN_FILE,
  data = stan_data,
  chains = CHAINS,
  iter = ITER,
  warmup = WARMUP,
  cores = CORES,
  seed = SEED
)
cat("Tiempo:", format(Sys.time() - t0), "\n")
saveRDS(fit, file.path(OUT_DIR, "fit_instability.rds"))
write.csv(summary(fit)$summary, file.path(OUT_DIR, "summary_global.csv"))

# --- 4. EXTRACCIÓN ---
ext <- rstan::extract(fit)
codigos <- raw[, .(ordre = unique(ordre)), by = codi_parada][order(ordre)]$codi_parada

phi_mean <- colMeans(ext$phi)
sigma_mean <- colMeans(ext$sigma_eta)
amp_mean <- colMeans(ext$amp_cum)
irrecov_mean <- colMeans(ext$irrecov_index)
prob_amp <- colMeans(ext$prob_amp_gt1)
prob_phi_gt1 <- colMeans(ext$phi > 1)
sens_min <- colMeans(ext$sens_min_origen)

df_stop <- data.frame(
  ordre = 2:S_real,
  parada = codigos[2:S_real],
  phi = phi_mean,
  sigma_min = sigma_mean,
  amp_cum = amp_mean,
  irrecov_index = irrecov_mean,
  prob_amp_cum_gt1 = prob_amp,
  prob_phi_gt1 = prob_phi_gt1,
  sens_min_por_min_origen = sens_min
)
df_stop <- df_stop[order(-df_stop$irrecov_index), ]

# --- 5. FIGURA ÚNICA (A4 / LaTeX) ---
# Boxplot headway × franja × tipo de día, replicado en las 39 paradas
raw_plot <- copy(raw)
hora_by_trip <- setNames(hora_trip, trips$trip_id)
raw_plot[, hora_trip := hora_by_trip[trip_id]]
raw_plot[, tipo_dia := fifelse(weekend == 1, "Fin de semana", "Laborable")]
raw_plot[, franja := cut(
  hora_trip,
  breaks = seq(6, 22, by = 2),
  include.lowest = TRUE,
  labels = paste0(seq(6, 20, by = 2), "-", seq(8, 22, by = 2), "h")
)]
raw_plot[, parada_lbl := factor(ordre, levels = 1:S_real, labels = paste0("P", 1:S_real))]

emp_var <- raw_plot[, .(
  media_hw = mean(headway),
  sd_hw = sd(headway),
  n = .N
), by = .(ordre, franja, tipo_dia)]

theme_latex <- theme_bw(base_size = 9, base_family = "serif") +
  theme(
    plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 8, hjust = 0.5, colour = "grey30"),
    strip.text = element_text(size = 7, face = "bold"),
    strip.background = element_rect(fill = "grey92", colour = NA),
    axis.title = element_text(size = 9),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 6),
    axis.text.y = element_text(size = 6),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(0.35, "lines"),
    plot.margin = margin(6, 8, 6, 8)
  )

p_entregable <- ggplot(raw_plot, aes(x = franja, y = headway, fill = tipo_dia)) +
  geom_boxplot(
    outlier.size = 0.4,
    outlier.alpha = 0.35,
    linewidth = 0.35,
    position = position_dodge(width = 0.75)
  ) +
  scale_fill_manual(values = c("Laborable" = "#3274A1", "Fin de semana" = "#E1812C")) +
  scale_y_continuous(limits = c(0, 30), breaks = seq(0, 30, by = 10)) +
  facet_wrap(~ parada_lbl, ncol = 13, scales = "fixed") +
  labs(
    title = "Headway por franja horaria y tipo de día — línea H8 (Anada)",
    subtitle = paste0(
      "Cada panel: una parada (P1–P39). Franja según hora de salida en parada 1. ",
      FECHA_INICIO, " – ", FECHA_FIN, ". n = ", nrow(raw_plot), " obs."
    ),
    x = "Franja horaria",
    y = "Headway (min)"
  ) +
  theme_latex +
  guides(fill = guide_legend(nrow = 1))

fig_path <- file.path(OUT_DIR, "headway_paradas_hora_finde")
ggsave(
  paste0(fig_path, ".pdf"),
  p_entregable,
  width = 11.69, height = 8.27,   # A4 apaisado
  device = cairo_pdf
)
ggsave(
  paste0(fig_path, ".png"),
  p_entregable,
  width = 11.69, height = 8.27,
  dpi = 300,
  bg = "white"
)
cat("Figura guardada:", fig_path, ".pdf / .png\n")

# --- 6. RESUMEN ---
cat("\n=== INESTABILIDAD IRRECUPERABLE ===\n")
cat(sprintf("Headway origen (ref.): %.1f min | σ origen: %.2f min\n",
            mean(ext$alpha1), mean(ext$sigma1)))
cat(sprintf("Fin de semana origen: %+.1f min\n", mean(ext$beta_we)))
cat(sprintf("μ_φ: %.3f | σ convoy (platoon): %.2f min\n",
            mean(ext$mu_phi), if (N_platoon > 0) mean(ext$sigma_platoon) else 0))
cat(sprintf("Trayectos en convoy: %d / %d\n", n_en_tren, P))
cat("\nTop 5 paradas más inestables (irrecov_index):\n")
print(df_stop[1:5, c("ordre", "parada", "phi", "sigma_min", "irrecov_index", "prob_amp_cum_gt1")])

hora_peor <- emp_var[, .(sd = max(sd_hw, na.rm = TRUE)), by = franja][order(-sd)][1, franja]
cat(sprintf("\nFranja con mayor σ empírica (global): %s\n", hora_peor))

write.csv(df_stop, file.path(OUT_DIR, "ranking_inestabilidad_paradas_global.csv"), row.names = FALSE)
write.csv(emp_var, file.path(OUT_DIR, "variabilidad_empirica_hora_finde.csv"), row.names = FALSE)
emp_rank <- emp_var[order(-sd_hw)][seq_len(min(20L, nrow(emp_var)))]
write.csv(emp_rank, file.path(OUT_DIR, "ranking_inestabilidad_hora_finde.csv"), row.names = FALSE)

cat("\n=== Resultados en:", OUT_DIR, "===\n")
