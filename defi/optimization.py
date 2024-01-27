import numpy as np

from defi.algorithms import gradient_descent
from defi.returns import generate_market
from defi.utils import get_var_cvar_empcdf, compute_returns


def optimize_iterate(params, N_iterations):

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

    for i in range(N_iterations):

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

    return all_weights, all_returns, all_metrics

