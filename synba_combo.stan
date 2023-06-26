data {
  int<lower=0> N;
  int<lower=0> N_test;
  vector[N] X1;
  vector[N] X2;
  vector[N] Y;
  vector[N_test] X1_test;
  vector[N_test] X2_test;
  real<lower=0> e0_mean;
  real<lower=0> einf_beta_a;
  real<lower=0> einf_beta_b;
  real log_ic50_lb;
  real log_ic50_ub;
  real sigma_mu;
}
parameters {
  real<lower=0> e_0;
  real<lower=0, upper=1> e_1;
  real<lower=0, upper=1> e_2;
  real<lower=0, upper=1> e_3;
  real<lower=log_ic50_lb, upper=log_ic50_ub> logC_1;
  real<lower=log_ic50_lb, upper=log_ic50_ub> logC_2;
  real<lower=0> h_1;
  real<lower=0> h_2;
  real<lower=0> alpha;
  real<lower=0> sigma;
}
model {
    e_0 ~ normal(e0_mean, 0.03*e0_mean);
    e_1 ~ beta(einf_beta_a, einf_beta_b);
    e_2 ~ beta(einf_beta_a, einf_beta_b);
    e_3 ~ beta(einf_beta_a, einf_beta_b);
    h_1 ~ lognormal(0, 1);
    h_2 ~ lognormal(0, 1);
    alpha ~ lognormal(0, 1);
    sigma ~ lognormal(sigma_mu, 1);
    for(i in 1:N){
        Y[i] ~ normal(((exp(logC_1))^h_1 * (exp(logC_2))^h_2 * e_0 + X1[i]^h_1 * (exp(logC_2))^h_2 * e_1 * e_0 + X2[i]^h_2 * (exp(logC_1))^h_1 * e_2 * e_0 + alpha * X1[i]^h_1 * X2[i]^h_2 * e_3 * e_0) / ((exp(logC_1))^h_1 * (exp(logC_2))^h_2 + X1[i]^h_1 * (exp(logC_2))^h_2 + X2[i]^h_2 * (exp(logC_1))^h_1 + alpha * X1[i]^h_1 * X2[i]^h_2), sigma);
    }
}
generated quantities {
    vector[N_test] Y_synthetic;
    for(i in 1:N_test){
        Y_synthetic[i] = normal_rng(((exp(logC_1))^h_1 * (exp(logC_2))^h_2 * e_0 + X1_test[i]^h_1 * (exp(logC_2))^h_2 * e_1 * e_0 + X2_test[i]^h_2 * (exp(logC_1))^h_1 * e_2 * e_0 + alpha * X1_test[i]^h_1 * X2_test[i]^h_2 * e_3 * e_0) / ((exp(logC_1))^h_1 * (exp(logC_2))^h_2 + X1_test[i]^h_1 * (exp(logC_2))^h_2 + X2_test[i]^h_2 * (exp(logC_1))^h_1 + alpha * X1_test[i]^h_1 * X2_test[i]^h_2), sigma);
    }
}