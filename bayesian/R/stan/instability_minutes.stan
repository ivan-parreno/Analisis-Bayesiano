// instability_minutes.stan
// Headways en MINUTOS. Pregunta: ¿qué parada / hora genera variabilidad
// irrecuperable aguas abajo?
//
// Unidad: trayecto = par_id × fecha (recorrido de un par delante→detrás)
// Platoon: pares encadenados (bus_detras de A = bus_delante de B, mismo día)
//   platoon_idx = 0 → par aislado (sin efecto de tren)
//   platoon_idx > 0 → comparte shock de tren con otros pares del convoy
//
// Inestabilidad irrecuperable:
//   φ[s] > 1 → amplifica desvíos (no converge)
//   amp_cum[s] = ∏φ → si > 1, perturbaciones crecen hacia el final
//   irrecov[s] = amp_cum[s] × σ_η[s] → magnitud combinada

data {
  int<lower=1> P;
  int<lower=1> S;
  array[P] int<lower=1, upper=S> N_stops;
  int<lower=1> N_total;

  array[N_total] int<lower=1, upper=S> stop;
  array[N_total] real<lower=1, upper=30> y;   // headway en MINUTOS
  array[N_total] int<lower=0, upper=1> weekend;

  array[P + 1] int<lower=1> pos;
  vector[P] hora_trip;

  int<lower=1> K;
  matrix[P, K] B_trip;

  // Trenes de buses encadenados (0 = par suelto, sin efecto platoon)
  int<lower=0> N_platoon;
  array[P] int<lower=0, upper=N_platoon> platoon_idx;
}

parameters {
  real<lower=1, upper=25> alpha1;       // headway medio origen (min), hora ref.
  real                  beta_we;        // min extra fin de semana en origen
  vector[K]             w_spline;       // minutos: efecto hora en origen
  real<lower=0.1>       sigma1;         // σ origen (min)

  real                  mu_alpha;       // salto medio por tramo (min)
  real<lower=0>         sigma_alpha;
  real<lower=0, upper=2> mu_phi;
  real<lower=0>         sigma_phi;
  real                  beta_we_drift;
  real<lower=0.1>       scale_sigma_eta;

  vector[S - 1]         alpha_inc_raw;
  array[S - 1]          real<lower=0, upper=2> phi;
  vector<lower=0.1>[S - 1] sigma_eta;

  real<lower=0>         sigma_platoon;  // shock compartido del convoy (min)
  vector[N_platoon]     z_platoon_raw;  // NC, solo si N_platoon > 0
}

transformed parameters {
  vector[S - 1] alpha_inc;
  vector[N_platoon] z_platoon;

  alpha_inc = mu_alpha + sigma_alpha * alpha_inc_raw;
  if (N_platoon > 0) {
    z_platoon = sigma_platoon * z_platoon_raw;
  }
}

model {
  mu_alpha        ~ normal(0, 1);
  sigma_alpha     ~ exponential(2);
  mu_phi          ~ normal(1, 0.15);
  sigma_phi       ~ exponential(2);     // menos shrinkage → paradas más distintas
  beta_we_drift   ~ normal(0, 0.5);
  scale_sigma_eta ~ exponential(1);

  alpha1          ~ normal(12, 4);
  beta_we         ~ normal(0, 2);
  w_spline        ~ normal(0, 1.5);
  sigma1          ~ normal(2, 1);

  alpha_inc_raw   ~ std_normal();
  for (s in 1:(S - 1)) {
    phi[s]       ~ normal(mu_phi, sigma_phi);
    sigma_eta[s] ~ normal(0, scale_sigma_eta);
  }

  if (N_platoon > 0) {
    sigma_platoon ~ exponential(1);
    z_platoon_raw ~ std_normal();
  }

  for (p in 1:P) {
    int i_start = pos[p];
    int n       = N_stops[p];
    real plat   = (platoon_idx[p] > 0) ? z_platoon[platoon_idx[p]] : 0.0;
    real mu1    = alpha1 + beta_we * weekend[i_start]
                + dot_product(B_trip[p], w_spline) + plat;

    mu1 = fmin(30.0, fmax(1.0, mu1));
    y[i_start] ~ normal(mu1, sigma1);

    if (n > 1) {
      for (j in 2:n) {
        int curr  = i_start + j - 1;
        int prev  = curr - 1;
        int s_idx = stop[curr] - 1;
        real mu   = alpha_inc[s_idx]
                  + beta_we_drift * weekend[curr]
                  + phi[s_idx] * y[prev]
                  + plat;
        mu = fmin(30.0, fmax(0.5, mu));

        y[curr] ~ normal(mu, sigma_eta[s_idx]);
      }
    }
  }
}

generated quantities {
  vector[N_total] log_lik;

  // Amplificación acumulada desde origen (φ>1 → inestabilidad irrecuperable)
  array[S - 1] real amp_cum;
  array[S - 1] real irrecov_index;
  array[S - 1] real prob_amp_gt1;

  amp_cum[1] = phi[1];
  irrecov_index[1] = amp_cum[1] * sigma_eta[1];
  prob_amp_gt1[1] = amp_cum[1] > 1.0 ? 1.0 : 0.0;

  for (s in 2:(S - 1)) {
    amp_cum[s] = amp_cum[s - 1] * phi[s];
    irrecov_index[s] = amp_cum[s] * sigma_eta[s];
    prob_amp_gt1[s] = amp_cum[s] > 1.0 ? 1.0 : 0.0;
  }

  // Sensibilidad en minutos: 1 min extra en origen → amp_cum[s] min extra esperados
  array[S - 1] real sens_min_origen;
  sens_min_origen = amp_cum;

  for (p in 1:P) {
    int i_start = pos[p];
    int n       = N_stops[p];
    real plat   = (platoon_idx[p] > 0) ? z_platoon[platoon_idx[p]] : 0.0;
    real mu1    = alpha1 + beta_we * weekend[i_start]
                + dot_product(B_trip[p], w_spline) + plat;
    mu1 = fmin(30.0, fmax(1.0, mu1));

    log_lik[i_start] = normal_lpdf(y[i_start] | mu1, sigma1);

    if (n > 1) {
      for (j in 2:n) {
        int curr  = i_start + j - 1;
        int prev  = curr - 1;
        int s_idx = stop[curr] - 1;
        real mu   = alpha_inc[s_idx]
                  + beta_we_drift * weekend[curr]
                  + phi[s_idx] * y[prev]
                  + plat;
        mu = fmin(30.0, fmax(0.5, mu));

        log_lik[curr] = normal_lpdf(y[curr] | mu, sigma_eta[s_idx]);
      }
    }
  }
}
