data {
  int<lower=0> N;
  int<lower=0> N_test;
  vector[N] X;
  vector[N] Y;
  vector[N_test] X_test;
  real<lower=0> e0_mean;
  real<lower=0> einf_beta_a;
  real<lower=0> einf_beta_b;
  real log_ic50_lb;
  real log_ic50_ub;
  real sigma_mu;
}
parameters {
  real<lower=0> e0;
  real<lower=0, upper=1> einf;
  real<lower=log_ic50_lb, upper=log_ic50_ub> logC;
  real<lower=0> h;
  real<lower=0> sigma;
}
model {
    h ~ lognormal(0, 1);
    einf ~ beta(einf_beta_a, einf_beta_b);
    sigma ~ lognormal(sigma_mu, 1);
    e0 ~ normal(e0_mean, 0.03*e0_mean);
    for(i in 1:N)
    Y[i] ~ normal(e0 + (einf * e0 - e0) / (1.0 + (exp(logC) / X[i]) ^ h), sigma);
}
generated quantities {
    vector[N_test] Y_synthetic;
    for(i in 1:N_test){
        Y_synthetic[i] = normal_rng(e0 + (einf * e0 - e0) / (1.0 + (exp(logC) / X_test[i]) ^ h), sigma);
    }
}