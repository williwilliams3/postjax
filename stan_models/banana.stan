data {
    vector[100] y;
}
parameters {
    vector[2] theta;
}
model {
    theta ~ normal(0, 2);
    y ~ normal(theta[1] + square(theta[2]), 2);
}