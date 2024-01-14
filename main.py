# -*- coding: utf-8 -*-

from amm import amm
import numpy as np
from scipy.stats import norm
import riskfolio as rp
import pandas as pd

import matplotlib.pyplot as plt
# plt.style.use('paper.mplstyle')

import seaborn as sns

sns.set_theme(style="ticks")

import statsmodels.api as sm
import matplotlib.ticker as mtick
import pickle
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':

    # Generate returns

    T = 60
    Rx = 100.
    Ry = 1000.
    batch_size = 32

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

    # Building the portfolio object
    port = rp.Portfolio(returns=Y)

    # Calculating optimal portfolio

    # Select method and estimate input parameters:

    method_mu = 'hist'  # Method to estimate expected returns based on historical data.
    method_cov = 'hist'  # Method to estimate covariance matrix based on historical data.

    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

    # Estimate optimal portfolio:

    model = 'Classic'  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
    rm = 'CVaR'  # Risk measure used, this time will be variance
    obj = 'MinRisk'  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist = True  # Use historical scenarios for risk measures that depend on scenarios
    rf = 0  # Risk free rate
    l = 0  # Risk aversion factor, only useful when obj is 'Utility'

    weights = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

    print("Portfolio weights:")

    portfolio_returns = log_ret @ weights
    portfolio_returns = portfolio_returns.to_numpy().T[0]
