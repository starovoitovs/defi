import numpy as np


def get_var_cvar_empcdf(returns, alpha, zeta):
    qtl = np.quantile(-returns, alpha)
    cvar = np.mean(-returns[-returns >= qtl])
    empcdf = np.where(~np.any(np.isnan(returns), axis=0), np.mean(returns >= zeta, axis=0), np.nan)
    return qtl, cvar, empcdf


def compute_returns(xs, ys, rxs, rys, phis, x_0):
    x_burn = np.sum(xs, axis=1)
    y_burn = np.sum(ys, axis=1)
    x_swap = y_burn * (1 - phis) * rxs / (rys + (1 - phis) * y_burn)
    return np.log(x_burn + x_swap) - np.log(x_0)
