"""Time-varying extremes: replication of Müller & Watson (2024) and extensions."""

from .likelihood import (
    gev_logpdf, gev_loglik, gev_score,
    exceedance_logpdf, exceedance_loglik, exceedance_score,
    gev_quantile, gev_to_pgp, pgp_to_gev, gp_to_gev, tvp_to_gev, h_extended,
)
from .mle import FitResult, fit_gev, fit_exceedance, lr_statistic
