import cvxpy as cp
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from amm import amm
import numpy as np
from scipy.linalg import sqrtm
import seaborn as sns
from scipy.stats import norm

from utils import plot_hist, plot_2d

sns.set_theme(style="ticks")

import pandas as pd

if __name__ == '__main__':

    # Generate returns

    T = 60
    Rx = 100.
    Ry = 1000.
    batch_size = 1_000

    kappa = [0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    sigma = [1, 0.3, 0.5, 1.5, 1.75, 2, 2.25]
    p = [0.35, 0.3, 0.34, 0.33, 0.32, 0.31, 0.3]
    phi = np.repeat(0.03, 7)

    ''' Initial reserves '''
    Rx0 = np.repeat(Rx, len(kappa))
    Ry0 = np.repeat(Ry, len(kappa))

    pools = amm(Rx=Rx0, Ry=Ry0, phi=phi)

    xs_0 = np.repeat(1., len(kappa))
    l = pools.swap_and_mint(xs_0)
    end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = pools.simulate(kappa, p, sigma, T=T, batch_size=batch_size)

    x_T = np.array([pool.burn_and_swap(l) for pool in end_pools])
    log_ret = np.log(x_T)
    Y = pd.DataFrame(log_ret)

    # 1d marginals on top of each other

    for i in range(log_ret.shape[1]):
        sns.kdeplot(log_ret[:, i], label=i)

    plt.xlim((-0.2, 0.2))
    plt.legend()
    plt.show()

    # 2d plot of returns
    plot_2d(log_ret)

    # pseudocode:
    # reference file: riskfolio/src/Portfolio.py

    returns = log_ret
    alpha = 0.05
    zeta = 0.7

    n_returns, n_assets = returns.shape

    weights = cp.Variable((n_assets,))
    X = returns @ weights

    Z = cp.Variable((n_returns,))
    var = cp.Variable((1,))
    cvar = var + 1 / (alpha * n_returns) * cp.sum(Z)

    constraints = [cp.sum(weights) == 1., weights <= 1., weights * 1000 >= 0]

    # CVaR constraints
    constraints += [Z * 1000 >= 0, Z * 1000 >= -X * 1000 - var * 1000]

    # # lower bound: average of emp cdf:
    # # might not be a valid constraint!
    # emp_cdf_005 = np.mean(log_ret >= 0.05, axis=0)
    # constraints += [emp_cdf_005 @ weights * 1000 >= zeta * 1000]

    # normal approximation with SOC constraint
    mean, cov = log_ret.mean(axis=0), np.cov(log_ret.T)
    sqrtcov = sqrtm(cov)
    constraints += [cp.SOC((-0.05 + mean @ weights) / norm.ppf(zeta), sqrtcov @ weights)]

    objective = cp.Minimize(cvar * 1000)

    # possible solvers: "ECOS", "SCS", "OSQP", "CVXOPT"
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver="SCS")

    print(f"Objective: {result}")

    portfolio_weights = weights.value
    portfolio_returns = returns @ portfolio_weights

    print("Portfolio weights:")
    display(pd.DataFrame(portfolio_weights).T)
    plot_hist(portfolio_returns)
