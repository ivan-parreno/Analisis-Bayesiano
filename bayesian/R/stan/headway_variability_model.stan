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
  real beta;
  
  vector[S] z_stop_mu;
  vector[Tt] z_time_mu;
  
  vector[S] z_stop_kappa;
  vector[Tt] z_time_kappa;
  
  real<lower=0> sigma_stop_mu;
  real<lower=0> sigma_time_mu;
  
  real<lower=0> sigma_stop_kappa;
  real<lower=0> sigma_time_kappa;
}

transformed parameters {
  vector[S] a_stop_mu;
  vector[Tt] b_time_mu;
  
  vector[S] a_stop_kappa;
  vector[Tt] b_time_kappa;
  
  vector[N] mu;
  vector[N] kappa;
  
  a_stop_mu = sigma_stop_mu * z_stop_mu;
  b_time_mu = sigma_time_mu * z_time_mu;
  
  a_stop_kappa = sigma_stop_kappa * z_stop_kappa;
  b_time_kappa = sigma_time_kappa * z_time_kappa;
  
  for (n in 1:N) {
    mu[n] = exp(alpha + a_stop_mu[stop[n]] + b_time_mu[time[n]]);
    kappa[n] = exp(beta + a_stop_kappa[stop[n]] + b_time_kappa[time[n]]);
  }
}

model {
  alpha ~ normal(log(mean(H)), 2);
  beta ~ normal(log(5), 1);
  
  z_stop_mu ~ normal(0, 1);
  z_time_mu ~ normal(0, 1);
  
  z_stop_kappa ~ normal(0, 1);
  z_time_kappa ~ normal(0, 1);
  
  sigma_stop_mu ~ normal(0, 0.5);
  sigma_time_mu ~ normal(0, 0.5);
  
  sigma_stop_kappa ~ normal(0, 0.5);
  sigma_time_kappa ~ normal(0, 0.5);
  
  H ~ gamma(kappa, kappa ./ mu);
}

generated quantities {
  vector[N] H_rep;
  vector[N] CV;
  
  for (n in 1:N) {
    H_rep[n] = gamma_rng(kappa[n], kappa[n] / mu[n]);
    CV[n] = inv_sqrt(kappa[n]);
  }
}