import logging
import os
import time

import numpy as np

from defi.algorithms import gradient_descent
from defi.returns import generate_market
from defi.utils import get_var_cvar_empcdf, compute_returns


def optimize(params):

    start_time = time.time()
    logging.info(f"Starting `test_optimize`.")

    weights = params['weights'].copy()

    all_weights = np.zeros((0, params['N_pools']))
    all_returns = np.zeros((0, params['batch_size']))
    all_metrics = np.zeros((0, 3))

    # generate and store initial returns
    xs, ys, rxs, rys, phis = generate_market(params)

    portfolio_returns = compute_returns(xs, ys, rxs, rys, phis, params['x_0'])
    var, cvar_actual, empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])

    all_weights = np.vstack([all_weights, weights.reshape(1, -1)])
    all_returns = np.vstack([all_returns, portfolio_returns.reshape(1, -1)])
    all_metrics = np.vstack([all_metrics, np.array([[np.nan, cvar_actual, empcdf]])])

    logging.info(np.array([cvar_actual, empcdf, *weights]))

    for i in range(params['N_iterations_refinement']):

        weights, cvar_algorithm = gradient_descent(xs, ys, rxs, rys, phis, {**params, 'weights': weights})

        # need this, because zero submission into pools is not allowed
        weights = params['weight_eps'] + weights
        weights /= np.sum(weights)

        # generate new returns with new weights for the next iteration
        xs, ys, rxs, rys, phis = generate_market({**params, 'weights': weights})

        portfolio_returns = compute_returns(xs, ys, rxs, rys, phis, params['x_0'])
        var, cvar_actual, empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])

        all_weights = np.vstack([all_weights, weights.reshape(1, -1)])
        all_returns = np.vstack([all_returns, portfolio_returns.reshape(1, -1)])
        all_metrics = np.vstack([all_metrics, np.array([[cvar_algorithm, cvar_actual, empcdf]])])

        logging.info(np.array([cvar_actual, empcdf, *weights]))

    os.makedirs(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy'), exist_ok=True)
    np.save(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy', 'weights.npy'), all_weights)
    np.save(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy', 'returns.npy'), all_returns)
    np.save(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy', 'metrics.npy'), all_metrics)

    end_time = time.time() - start_time
    logging.info(f"Successfully finished `test_optimize` after {end_time:.3f} seconds.")

    return all_weights, all_returns, all_metrics

