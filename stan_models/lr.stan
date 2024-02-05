data {
  int<lower=0> N;
  int<lower=0> D;
  matrix[N, D] x;
  int<lower=0,upper=1> y[N];
}
parameters {
  vector[D] beta;
}
model {
  beta ~ normal(0, 10);
  y ~ bernoulli_logit(x * beta);
}