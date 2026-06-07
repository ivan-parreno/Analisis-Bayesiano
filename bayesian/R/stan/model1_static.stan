// Modelo 1 — Reloj Estático (H8 · Anada)
// log(mu_i) = alpha + a[s] + b[h] + c[d]
//   a[s] ~ N(0, sigma_s),  c[d] ~ N(0, sigma_d)
//   b[h] Gaussian Random Walk con paso sigma_t
//   y_shifted ~ Gamma(kappa, kappa / mu)   [media mu, forma kappa global]

data {
  int<lower=0> N;
  vector<lower=0>[N] y;           // headway_min - 0.16
  array[N] int<lower=1> s;
  array[N] int<lower=1> h;
  array[N] int<lower=1> d;
  int<lower=1> n_s;
  int<lower=1> n_h;
  int<lower=1> n_d;
}

parameters {
  real alpha;
  vector[n_s] a_raw;
  vector[n_h] b_raw;
  vector[n_d] c_raw;
  real<lower=0> sigma_s;
  real<lower=0> sigma_t;
  real<lower=0> sigma_d;
  real<lower=0> kappa;
}

transformed parameters {
  vector[n_s] a = sigma_s * a_raw;
  vector[n_d] c = sigma_d * c_raw;
  vector[n_h] b;
  b[1] = sigma_t * b_raw[1];
  for (t in 2:n_h) {
    b[t] = b[t - 1] + sigma_t * b_raw[t];
  }
  vector[N] log_mu;
  for (i in 1:N) {
    log_mu[i] = alpha + a[s[i]] + b[h[i]] + c[d[i]];
  }
  vector[N] mu = exp(log_mu);
}

model {
  alpha ~ normal(2.0, 10.0);
  sigma_s ~ normal(0, 10.0);
  sigma_t ~ normal(0, 10.0);
  sigma_d ~ normal(0, 10.0);
  kappa ~ exponential(0.01);

  a_raw ~ std_normal();
  c_raw ~ std_normal();
  b_raw ~ std_normal();

  for (i in 1:N) {
    y[i] ~ gamma(kappa, kappa / mu[i]);
  }
}

generated quantities {
  matrix[n_s, n_h] log_mu_ref[n_d];
  for (dd in 1:n_d) {
    for (ss in 1:n_s) {
      for (hh in 1:n_h) {
        log_mu_ref[dd, ss, hh] = alpha + a[ss] + b[hh] + c[dd];
      }
    }
  }
}
