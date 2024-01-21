import json
import logging
import os
import sys
import time
from datetime import datetime
import numpy as np
import seaborn as sns

from defi.returns import generate_returns
from defi.utils import get_var_cvar_empcdf
from params import params
from defi.validation import test_algorithms, test_returns, test_market_impact

sns.set_theme(style="ticks")

additional_params = {
    # learning rate of the gradient descent
    'learning_rate_gd': 1e-2,
    # number of iterations for gradient descent
    'N_iterations_gd': 2000,
    # loss weights in gradient descent
    'loss_weights_gd': [1e3, 1., 1., 1.],
    # learning rate of the market impact model
    'learning_rate_mi': 5e-1,
    # number of iterations in market impact model
    'N_iterations_mi': 20,
    # minimum weight to ensure to we never put 0 coins in a pool
    'weight_eps': 1e-4,
    # initial weights used for generation of returns and as w0 for some algos, unless overridden
    'weights': np.repeat(1., params['N_pools']) / params['N_pools'],
}


def test1(params):
    # compute distributions and metrics for various weights
    # we consider the situation where one weight is being varied, while the others are equal, and so that they add up to 1
    # for example, [0.8, 0.1, 0.1]
    wis = np.linspace(params['weight_eps'], 1 - params['weight_eps'], 11)

    params_diffs = []

    for pool_tested in range(params['N_pools']):

        all_ws = []

        for wi in wis:
            ws = np.repeat(1. - wi, params['N_pools']) / (params['N_pools'] - 1)
            ws[pool_tested] = wi
            all_ws += [ws]

        params_diffs += [[{'weights': ws} for ws in all_ws]]

    xlabels = [f"w{pool_tested}" for pool_tested in range(params['N_pools'])]
    xaxes = [wis] * params['N_pools']
    test_returns(params, params_diffs, xlabels=xlabels, xaxes=xaxes)


def test2(params):
    # test algorithms for various values of q
    # we first generate returns and look at empirical cdf of the portfolio, in order to test the algorithms whenever chance constrained is attained
    returns = generate_returns(params)
    _, _, empcdf = get_var_cvar_empcdf(returns @ params['weights'], params['alpha'], params['zeta'])
    qs = np.linspace(max(empcdf - 0.08, 0.51), empcdf + 0.08, 21)
    test_algorithms(params, [{'q': q} for q in qs], xlabel="q", xaxis=qs)


def test3(params):
    # test market impact iteration
    test_market_impact(params)


if __name__ == '__main__':

    CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIRECTORY = os.path.join(f"_output/runs", CURRENT_TIMESTAMP)

    os.environ['RUN_ID'] = CURRENT_TIMESTAMP
    os.environ['OUTPUT_DIRECTORY'] = OUTPUT_DIRECTORY

    os.makedirs(os.environ['OUTPUT_DIRECTORY'], exist_ok=True)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'log.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )

    sys.stdout = logging.StreamHandler(sys.stdout)

    # merge original parameters and additional parameters
    params = {**params, **additional_params}

    # convert lists to ndarray
    params = {key: np.array(value) if isinstance(value, list) else value for key, value in params.items()}

    # dump params file
    with open(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'params.json'), 'w') as f:
        # convert ndarray to lists
        params_json = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in params.items()}
        json.dump(params_json, f, indent=4)

    # choose test case

    test1(params)
    test2(params)
    test3(params)
