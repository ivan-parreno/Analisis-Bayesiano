data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> Tt;

  array[N] int<lower=1, upper=S> stop;
  array[N] int<lower=1, upper=Tt> time;

  vector<lower=0>[N] H;
}

parameters {
  real alpha;

  vector[S] z_stop;
  vector[Tt] z_time;

  real<lower=0> sigma_stop;
  real<lower=0> sigma_time;

  real<lower=0> kappa;
}

transformed parameters {
  vector[S] a_stop;
  vector[Tt] b_time;
  vector[N] mu;

  a_stop = sigma_stop * z_stop;
  b_time = sigma_time * z_time;

  for (n in 1:N) {
    mu[n] = exp(alpha + a_stop[stop[n]] + b_time[time[n]]);
  }
}

model {
  alpha ~ normal(log(mean(H)), 2);

  z_stop ~ normal(0, 1);
  z_time ~ normal(0, 1);

  sigma_stop ~ normal(0, 0.5);
  sigma_time ~ normal(0, 0.5);

  kappa ~ exponential(0.5);

  for (n in 1:N) {
    H[n] ~ gamma(kappa, kappa / mu[n]);
  }
}

generated quantities {
  vector[N] H_rep;

  for (n in 1:N) {
    H_rep[n] = gamma_rng(kappa, kappa / mu[n]);
  }
}