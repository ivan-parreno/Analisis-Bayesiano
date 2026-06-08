# rstan libraries
library(tidyv)
library(dplyr)
library(rstan)
options(mc.cores = parallel::detectCores())

# impor dataset from the folder
df <- read.csv("eventos_nuevos/arrivals_2026-05-02_8_2_limpio_3.csv")

headways <- df %>%
  filter(!is.na(headway_min), headway_min > 0) %>%
  mutate(
    stop_id = as.integer(factor(paste(sentit, codi_parada))),
    time_id = as.integer(factor(hora_paso))
  )

stan_data <- list(
  N = nrow(headways),
  S = max(headways$stop_id),
  Tt = max(headways$time_id),
  stop = headways$stop_id,
  time = headways$time_id,
  H = headways$headway_min
)

fit <- stan(
  file = "models_stan/headway_model.stan",
  data = stan_data,
  chains = 4,
  cores = parallel::detectCores(),
  iter = 2000,
  warmup = 1000,
  seed = 123,
  control = list(adapt_delta = 0.95)
)


print(fit, pars = c("alpha", "sigma_stop", "sigma_time", "kappa"))

summary(fit, pars = c("alpha", "sigma_stop", "sigma_time", "kappa"))$summary

length(unique(headways$stop_id))
length(unique(headways$time_id))


# example of prediction for random stop and time
post <- rstan::extract(fit)

mu_draws <- exp(
  post$alpha +
    post$a_stop[,15] +
    post$b_time[,8]
)
quantile(mu_draws, c(0.025,0.5,0.975))
