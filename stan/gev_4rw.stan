// Müller & Watson (2024) GEV model with four random walks (xi, alpha, s, m).
//
// Copied from GEV_4RW_Model.stan in the authors' replication archive
// (external/muller-watson-2024/files/Stan/Stan_Code). The likelihood, priors and
// parameterization are unchanged. Only the output differs: the derived paths are local to the
// model block (so they are not written for every draw), and generated quantities write just the
// GEV paths (mu, sigma, xi) and the annual-maximum quantile paths q[p, t] for the levels in
// `probs`. This keeps the CSV output small when the model is fitted many times in a Monte Carlo
// study. The sampler sees exactly the same target density as the authors' model.
//
// Data y[j, t]: the k = nobs largest values of period t in decreasing order (y[1, t] largest).

functions {

  real GEVcpdf_lpdf(real x, real mu, real sig, real xsi){
    real z=(x-mu)/sig;
    real tau=exp(-6.0*abs(xsi));
    real lht;
    if(1+xsi*z<tau){
      lht=-log(tau)/xsi-(z-(tau-1)/xsi)/tau;
    }
    else{
      lht=-log1p(xsi*z)/xsi;
    }
    return (1+xsi)*lht-log(sig);
  }

  real GEVpdf_lpdf(real x, real mu, real sig, real xsi){
    real z=(x-mu)/sig;
    real tau=exp(-6.0*abs(xsi));
    real lht;
    if(1+xsi*z<tau){
      lht=-log(tau)/xsi-(z-(tau-1)/xsi)/tau;
    }
    else{
      lht=-log1p(xsi*z)/xsi;
    }
    return (1+xsi)*lht-log(sig)-exp(lht);
  }

  // GEV parameter paths from the model's parameters (as in the authors' transformed parameters)
  matrix gev_paths(int T, real xi_min, real xi_max, real sg_xi, real sg_alpha, real sg_s,
                   real trans_xi_level, real ln_s_level, real m_level,
                   real g_xi, real g_alpha, real g_s, real g_m,
                   vector h_xi, vector h_alpha, vector h_s, vector h_m) {
    real sg_m = exp(ln_s_level);
    real gamma_xi = sg_xi*g_xi/sqrt(T);
    real gamma_alpha = sg_alpha*g_alpha/sqrt(T);
    real gamma_s = sg_s*g_s/sqrt(T);
    real gamma_m = sg_m*g_m/sqrt(T);
    vector[T] c_xi = cumulative_sum(gamma_xi*h_xi);
    vector[T] c_alpha = cumulative_sum(gamma_alpha*h_alpha);
    vector[T] c_s = cumulative_sum(gamma_s*h_s);
    vector[T] c_m = cumulative_sum(gamma_m*h_m);
    vector[T] trans_xi = trans_xi_level + c_xi - mean(c_xi);
    vector[T] ln_alpha = c_alpha - mean(c_alpha);
    vector[T] ln_s = ln_s_level + c_s - mean(c_s);
    vector[T] m = m_level + c_m - mean(c_m);
    vector[T] alpha = exp(ln_alpha);
    vector[T] s = exp(ln_s);
    vector[T] xi = xi_min + (xi_max-xi_min)*(exp(trans_xi)./(1+exp(trans_xi)));
    matrix[T, 3] out;
    out[, 1] = m + (s./xi).*((alpha.^xi)-1);      // mu
    out[, 2] = s.*(alpha.^xi);                     // sigma
    out[, 3] = xi;
    return out;
  }

}

data {

    real xi_min;
    real xi_max;
    int<lower=0> T;
    int<lower=0> nobs;
    matrix[nobs,T] y;
    real sg_xi;
    real sg_alpha;
    real sg_s;
    int<lower=0> np;
    vector[np] probs;

}

parameters {

    real trans_xi_level;
    real ln_s_level;
    real m_level;
    real<lower=0.0,upper=5.0> g_xi;
    real<lower=0.0,upper=5.0> g_alpha;
    real<lower=0.0,upper=5.0> g_s;
    real<lower=0.0,upper=5.0> g_m;
    vector [T] h_xi;
    vector [T] h_alpha;
    vector [T] h_s;
    vector [T] h_m;

}

model {

    matrix[T, 3] P = gev_paths(T, xi_min, xi_max, sg_xi, sg_alpha, sg_s, trans_xi_level,
                               ln_s_level, m_level, g_xi, g_alpha, g_s, g_m,
                               h_xi, h_alpha, h_s, h_m);

    g_xi ~ exponential(1.0);
    g_alpha ~ exponential(1.0);
    g_s ~ exponential(1.0);
    g_m ~ exponential(1.0);

    trans_xi_level ~ normal(0.0,1.5);
    // Flat prior on ln_s_level and m_level

    h_xi ~ std_normal();
    h_alpha ~ std_normal();
    h_s ~ std_normal();
    h_m ~ std_normal();

    for (t in 1:T) {
      for (j in 1:nobs-1) {
        y[j,t] ~ GEVcpdf(P[t,1],P[t,2],P[t,3]);
      }
      y[nobs,t] ~ GEVpdf(P[t,1],P[t,2],P[t,3]);
    }

}

generated quantities {

    vector[T] mu;
    vector[T] sigma;
    vector[T] xi;
    matrix[np, T] q;          // annual-maximum quantiles: P(max <= q[i, t]) = probs[i]
    {
      matrix[T, 3] P = gev_paths(T, xi_min, xi_max, sg_xi, sg_alpha, sg_s, trans_xi_level,
                                 ln_s_level, m_level, g_xi, g_alpha, g_s, g_m,
                                 h_xi, h_alpha, h_s, h_m);
      mu = P[, 1];
      sigma = P[, 2];
      xi = P[, 3];
    }
    for (i in 1:np) {
      real lp = -log(probs[i]);
      for (t in 1:T) {
        q[i, t] = mu[t] + sigma[t] * (pow(lp, -xi[t]) - 1) / xi[t];
      }
    }

}
