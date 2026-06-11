library(tidyverse)
library(rstan)

rstan_options(auto_write = TRUE)
options(mc.cores = 4)

leer_headway <- function(f) {
  
  fecha <- str_extract(
    basename(f),
    "\\d{4}-\\d{2}-\\d{2}"
  )
  
  read.csv(f) %>%
    mutate(
      source_file = basename(f),
      fecha = as.Date(fecha)
    )
}

Y_SHIFT <- 0.16
HW_MIN <- 1
HW_MAX <- 30

DATA_DIR <- "processed/headways/"

hora_a_franja <- function(hora) {
  ifelse(hora <= 6, 0,
         ifelse(hora <= 21, hora - 6, 16))
}

files_info <- tibble(
  file = list.files(
    DATA_DIR,
    pattern = "^headways_pares3_H8_Anada_\\d{4}-\\d{2}-\\d{2}.*\\.csv$",
    full.names = TRUE
  )
) %>%
  mutate(
    fecha = as.Date(str_extract(basename(file), "\\d{4}-\\d{2}-\\d{2}")),
    weekday = as.integer(format(fecha, "%u"))
  ) %>%
  arrange(fecha) %>%
  group_by(weekday) %>%
  slice_head(n = 2) %>%
  ungroup()

files <- files_info$file

df_all <- map_dfr(files, leer_headway)

headways <- df_all %>%
  filter(
    !is.na(headway_pair),
    headway_pair >= HW_MIN,
    headway_pair <= HW_MAX
  ) %>%
  mutate(
    weekday = as.integer(format(fecha, "%u")),
    franja = hora_a_franja(as.integer(franja_hora)),
    H = headway_pair - Y_SHIFT
  ) %>%
  filter(H > 0) %>%
  mutate(
    stop = as.integer(factor(codi_parada)),
    time = as.integer(factor(paste(weekday, franja, sep = "_")))
  )

stan_data <- list(
  N = nrow(headways),
  S = max(headways$stop),
  Tt = max(headways$time),
  stop = headways$stop,
  time = headways$time,
  H = headways$H
)

table(headways$fecha)
table(headways$weekday)
length(unique(headways$stop))
length(unique(headways$time))
summary(headways$H)

fit_var <- rstan::stan(
  file = "bayesian/R/stan/headway_variability_model.stan",
  data = stan_data,
  chains = 4,
  cores = 4,
  iter = 2000,
  warmup = 500,
  seed = 123,
  control = list(adapt_delta = 0.99),
  refresh = 100
)

print(
  fit_var,
  pars = c(
    "alpha",
    "beta",
    "sigma_stop_mu",
    "sigma_time_mu",
    "sigma_stop_kappa",
    "sigma_time_kappa"
  )
)

#dir.create("bayesian/R/stan/fits", recursive = TRUE, showWarnings = FALSE)

saveRDS(
  fit_var,
  file.path("bayesian/R/stan/fits", "headway_variability_model_fit.rds")
)
print(
  fit_var,
  pars = c(
    "sigma_stop_kappa",
    "sigma_time_kappa"
  )
)
