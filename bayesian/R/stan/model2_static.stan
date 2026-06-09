// Modelo 2 — Reloj Dinámico (H8 · Anada)
// log(mu_i) = alpha + a[s] + b[h] + c[d]
//   a[s] Gaussian Random Walk espacial (paradas ordenadas por recorrido)
//   b[h] Gaussian Random Walk temporal (franjas horarias)
//   c[d] ~ N(0, sigma_d)  efecto día de semana i.i.d.
//   y_shifted ~ Gamma(kappa, kappa / mu)   [media mu, forma kappa global]

data {
  int<lower=0> N;
  vector<lower=0>[N] y;
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
  // Random Walk espacial sobre paradas (ordenadas por posición en recorrido)
  vector[n_s] a;
  a[1] = sigma_s * a_raw[1];
  for (t in 2:n_s) a[t] = a[t - 1] + sigma_s * a_raw[t];

  // Random Walk temporal sobre franjas horarias
  vector[n_h] b;
  b[1] = sigma_t * b_raw[1];
  for (t in 2:n_h) b[t] = b[t - 1] + sigma_t * b_raw[t];

  // Efecto día de semana i.i.d.
  vector[n_d] c = sigma_d * c_raw;

  // Media por observación
  vector[N] mu;
  for (i in 1:N) {
    mu[i] = exp(alpha + a[s[i]] + b[h[i]] + c[d[i]]);
  }
}

model {
  alpha   ~ normal(2.0, 10.0);
  sigma_s ~ normal(0, 10.0);
  sigma_t ~ normal(0, 10.0);
  sigma_d ~ normal(0, 10.0);
  kappa   ~ exponential(0.01);

  a_raw ~ std_normal();
  b_raw ~ std_normal();
  c_raw ~ std_normal();

  for (i in 1:N) {
    y[i] ~ gamma(kappa, kappa / mu[i]);
  }
}
