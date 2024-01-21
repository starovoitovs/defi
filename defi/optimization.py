import logging

import numpy as np

from defi.algorithms import cvx, gradient_descent
from defi.returns import generate_returns
from defi.utils import get_var_cvar_empcdf


def optimize(returns, params):
    """
    Run several optimizers and store the output.
    :param returns:
    :param params:
    """
    algorithms = []

    # unconstrained, for demostration purposes
    algorithms += [{'func': cvx, 'kwargs': {'mode': 0}}]

    # lower bound: average of emp cdf: - might not be a valid constraint! unless VaR subadditive
    algorithms += [{'func': cvx, 'kwargs': {'mode': 1}}]

    # normal approximation with SOC constraint
    algorithms += [{'func': cvx, 'kwargs': {'mode': 2}}]

    # gradient descent with backprop
    algorithms += [{'func': gradient_descent, 'kwargs': {}}]

    results_weights = np.full((len(algorithms), params['N_pools']), np.nan)

    N_completed = 0

    for j, algorithm in enumerate(algorithms):

        try:

            func, kwargs = algorithm['func'], algorithm['kwargs']

            portfolio_weights, *_ = func(returns, params, **kwargs)

            # need this, because zero submission into pools is not allowed
            portfolio_weights = params['weight_eps'] + portfolio_weights
            portfolio_weights /= np.sum(portfolio_weights)

            results_weights[j, :] = portfolio_weights

            N_completed += 1

        except Exception as e:
            logging.info(f"Algorithm {j} failed: {e}")
            continue

    return results_weights, N_completed


def iterate(params, params_diffs, break_on_constraint_violation=False, update_returns=False, store_initial_weights_and_returns=False):
    """
    Run several algorithms with varying parameters.
    :param params: initial parameters
    :param params_diffs: list with parameter variations
    :param break_on_constraint_violation: if True, iteration breaks if some algorithms succeed but all violation the chance constraint
    :param update_returns: if True, will compute new returns with updated weights at the end of each iteration - used in market impact iteration
    :param store_initial_weights_and_returns: if True, will create additional point in the beginning of output arrays to store initial weights and returns - used in market impact iteration
    """
    # generate initial returns
    initial_returns = generate_returns(params)

    N_algorithms = 4

    # if set to True, add one column at the front to store original weights and returns
    if store_initial_weights_and_returns:
        all_returns = np.full((params['batch_size'], len(params_diffs) + 1, params['N_pools']), np.nan)
        all_returns[:, 0, :] = initial_returns
        all_weights = np.full((len(params_diffs) + 1, N_algorithms, params['N_pools']), np.nan)
        all_weights[0, :, :] = params['weights'][None, :]
        all_best_weights = np.full((len(params_diffs) + 1, params['N_pools']), np.nan)
        all_best_weights[0, :] = params['weights']
    else:
        all_returns = np.full((params['batch_size'], len(params_diffs), params['N_pools']), np.nan)
        all_weights = np.full((len(params_diffs), N_algorithms, params['N_pools']), np.nan)
        all_best_weights = np.full((len(params_diffs), params['N_pools']), np.nan)

    returns = initial_returns
    best_weights = params['weights'].copy()

    for i, diff in enumerate(params_diffs):

        results_weights, N_completed = optimize(returns, {**params, 'weights': best_weights, **diff})

        # if failed for all constraints apart from unconstrained, early stoppage
        if N_completed <= 1:
            logging.info(f"All algorithms failed (up to unconstrained cvxpy). Stopping at iteration {i + 1}/{len(params_diffs)}.")
            break

        # identify the best weights among all algorithms

        portfolio_returns = returns @ results_weights.T
        var, cvar, empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])

        # select weights where the algorithms has finished, constraint is satisfied and CVaR is best
        arr = np.where(empcdf >= params['q'], cvar, -np.inf)

        # if no algorithm satisfies the constraint, break
        if break_on_constraint_violation and np.all(arr == -np.inf):
            logging.info(f"Some algorithms completed successfully, but none satisfied the probabilistic constraint. Stopping at iteration {i + 1}/{len(params_diffs)}.")
            break

        idx = np.argmax(arr)
        best_weights += params['learning_rate_mi'] * (results_weights[idx] - best_weights)

        # if update_returns=True, generate new returns with new weights for the next iteration
        if update_returns:
            returns = generate_returns({**params, 'weights': best_weights})

        idx = i + store_initial_weights_and_returns
        all_best_weights[idx, :] = best_weights
        all_weights[idx, :, :] = results_weights
        all_returns[:, idx, :] = returns

    return all_weights, all_best_weights, all_returns
