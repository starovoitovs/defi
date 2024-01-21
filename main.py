import json
import logging
import os
import sys
import time
from datetime import datetime
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from defi.algorithms import gradient_descent, cvx
from defi.returns import generate_returns
from defi.utils import get_var_cvar_empcdf, plot_metric
from params import params

sns.set_theme(style="ticks")

additional_params = {
    # learning rate of the gradient descent
    'learning_rate_gd': 1e-2,
    # number of iterations for gradient descent
    'N_iterations_gd': 2000,
    # loss weights in gradient descent
    'loss_weights': [1e3, 1., 1., 1.],
    # learning rate of the market impact model
    'learning_rate_mi': 5e-1,
    # number of iterations in market impact model
    'N_iterations_mi': 10,
}


def optimize(returns, params, initial_weights):
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

            portfolio_weights, *_ = func(returns, params, weights=initial_weights, **kwargs)

            # need this, because zero submission into pools is not allowed
            eps = 1e-8
            portfolio_weights = eps + portfolio_weights
            portfolio_weights /= np.sum(portfolio_weights)

            results_weights[j, :] = portfolio_weights

            N_completed += 1

        except Exception as e:
            logging.info(f"Algorithm {j} failed: {e}")
            continue

    return results_weights, N_completed


def iterate(params, diffs, break_on_constraint_violation=False, update_returns=False, store_initial_weights_and_returns=False):
    # always start with uniformly distributed portfolio
    initial_weights = np.repeat(1., params['N_pools']) / params['N_pools']
    initial_returns = generate_returns(params, initial_weights)

    N_algorithms = 4

    if store_initial_weights_and_returns:
        all_returns = np.full((params['batch_size'], len(diffs) + 1, params['N_pools']), np.nan)
        all_returns[:, 0, :] = initial_returns
        all_weights = np.full((len(diffs) + 1, N_algorithms, params['N_pools']), np.nan)
        all_weights[0, :, :] = initial_weights[None, :]
        all_best_weights = np.full((len(diffs) + 1, params['N_pools']), np.nan)
        all_best_weights[0, :] = initial_weights
    else:
        all_returns = np.full((params['batch_size'], len(diffs), params['N_pools']), np.nan)
        all_weights = np.full((len(diffs), N_algorithms, params['N_pools']), np.nan)
        all_best_weights = np.full((len(diffs), params['N_pools']), np.nan)

    returns = initial_returns
    best_weights = initial_weights

    for i, diff in enumerate(diffs):

        results_weights, N_completed = optimize(returns, {**params, **diff}, best_weights)

        # if failed for all constraints apart from unconstrained, early stoppage
        if N_completed <= 1:
            logging.info(f"All algorithms failed (up to unconstrained cvxpy). Stopping at iteration {i + 1}/{len(diffs)}.")
            break

        # identify the best weights among all algorithms

        portfolio_returns = returns @ results_weights.T
        var, cvar, empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])

        # select weights where the algorithms has finished, constraint is satisfied and CVaR is best
        arr = np.where(empcdf >= params['q'], cvar, -np.inf)

        # if no algorithm satisfies the constraint, break
        if break_on_constraint_violation and np.all(arr == -np.inf):
            logging.info(f"Some algorithms completed successfully, but none satisfied the probabilistic constraint. Stopping at iteration {i + 1}/{len(diffs)}.")
            break

        idx = np.argmax(arr)
        best_weights += params['learning_rate_mi'] * (results_weights[idx] - best_weights)

        # if update_returns=True, generate new returns with new weights for the next iteration
        if update_returns:
            returns = generate_returns(params, best_weights)

        idx = i + store_initial_weights_and_returns
        all_best_weights[idx, :] = best_weights
        all_weights[idx, :, :] = results_weights
        all_returns[:, idx, :] = returns

    return all_weights, all_best_weights, all_returns


def test_optimizers():
    # optimization test for various values of q
    qs = np.linspace(0.55, 0.65, 11)
    all_weights, all_best_weights, all_returns = iterate(params, [{'q': q} for q in qs],
                                                         break_on_constraint_violation=False,
                                                         store_initial_weights_and_returns=False,
                                                         update_returns=False)

    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_weights.npy'), all_weights)
    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_best_weights.npy'), all_best_weights)
    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_returns.npy'), all_returns)

    # compute returns and metrics: (N_return, N_diff, N_algorithm)
    portfolio_returns = np.einsum('jik,ilk->jil', all_returns, all_weights)
    _, portfolio_cvar, portfolio_empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])

    # plot
    fig, ax = plt.subplots(ncols=4, figsize=(24, 4), gridspec_kw={'hspace': 0.5})
    fig.subplots_adjust(bottom=0.2)

    bar_width = 0.15
    initial_returns = all_returns[:, 0, :]  # when testing optimizers, returns remain the same, so can take e.g. 0th ones
    _, marginal_cvar, marginal_empcdf = get_var_cvar_empcdf(initial_returns, params['alpha'], params['zeta'])

    pool_range = np.arange(initial_returns.shape[1])
    ax[0].bar(pool_range, np.mean(initial_returns, axis=0), width=bar_width, label='Mean')
    ax[0].bar(bar_width + pool_range, np.std(initial_returns, axis=0), width=bar_width, label='Std')
    ax[0].bar(2 * bar_width + pool_range, marginal_cvar, width=bar_width, label='Marginal CVaR')
    ax[0].bar(3 * bar_width + pool_range, marginal_empcdf, width=bar_width, label='Marginal $P(r \geq \zeta)$')
    ax[0].legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax[0].set_xlabel('Pool')
    ax[0].set_title("Properties of initial returns")

    line_labels = 'unconstrained', 'average cdf constraint', 'gaussian constraint', 'backprop'
    plot_metric(ax[1], qs, all_weights[:, :, 0], xlabel='q', title='Weight of pool0')
    plot_metric(ax[2], qs, portfolio_cvar, xlabel='q', title='Portfolio CVaR')
    plot_metric(ax[3], qs, portfolio_empcdf, line_labels, xlabel='q', title='$P(r \geq \zeta)$')

    # plot an extra line to visualize the constraint
    xmin, xmax = ax[3].get_xlim()
    qs_plot = np.linspace(xmin, xmax, 101)
    ax[3].plot(qs_plot, qs_plot, ls='--')

    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'optimization_metrics.pdf'))


def test_market_impact():
    iterations = np.arange(params['N_iterations_mi'])
    all_weights, all_best_weights, all_returns = iterate(params, [{} for _ in iterations],
                                                         break_on_constraint_violation=True,
                                                         store_initial_weights_and_returns=True,
                                                         update_returns=True)

    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_weights.npy'), all_weights)
    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_best_weights.npy'), all_best_weights)
    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_returns.npy'), all_returns)

    # compute returns and metrics: (N_return, N_diff, N_algorithm)
    portfolio_returns = np.einsum('jik,ilk->jil', all_returns, all_weights)
    _, portfolio_cvar, portfolio_empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])
    _, marginal_cvar, marginal_empcdf = get_var_cvar_empcdf(all_returns, params['alpha'], params['zeta'])

    # plots
    iterations = np.arange(params['N_iterations_mi'] + 1)
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 8), gridspec_kw={'hspace': 0.5})
    line_labels = 'unconstrained', 'average cdf constraint', 'gaussian constraint', 'backprop'
    plot_metric(ax[0][0], iterations, all_weights[:, :, 0], xlabel='N_iterations', title='Weight of pool0')
    plot_metric(ax[0][1], iterations, portfolio_cvar, xlabel='N_iterations', title='Portfolio CVaR')
    plot_metric(ax[0][2], iterations, portfolio_empcdf, xlabel='N_iterations', title='$P(r \geq \zeta)$')

    # plot an extra line to visualize the constraint q
    xmin, xmax = ax[0][2].get_xlim()
    ax[0][2].hlines(params['q'], xmin, xmax, color='k', linestyles='--')

    plot_metric(ax[0][3], iterations, np.mean(portfolio_returns, axis=0), line_labels=line_labels, xlabel='N_iterations', title='Mean return')

    # plot return characteristics as function of iteration
    line_labels = [f"pool{i}" for i in range(params['N_pools'])]
    plot_metric(ax[1][0], iterations, all_best_weights, xlabel='N_iterations', title='Best weights')
    plot_metric(ax[1][1], iterations, marginal_cvar, xlabel='N_iterations', title='Marginal CVaR')
    plot_metric(ax[1][2], iterations, marginal_empcdf, xlabel='N_iterations', title='Marginal $P(r_i \geq \zeta)$')

    # plot an extra line to visualize the constraint q
    xmin, xmax = ax[1][2].get_xlim()
    ax[1][2].hlines(params['q'], xmin, xmax, color='k', linestyles='--')

    plot_metric(ax[1][3], iterations, np.mean(all_returns, axis=0), line_labels=line_labels, xlabel='N_iterations', title='Marginal Mean')

    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'optimization_metrics.pdf'))


if __name__ == '__main__':
    start_time = time.time()

    PARENT_DIRECTORY = f"_output/runs"
    RUN_DIRECTORY = datetime.now().strftime("%Y%d%m_%H%M%S")
    OUTPUT_DIRECTORY = os.path.join(PARENT_DIRECTORY, RUN_DIRECTORY)

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIRECTORY, 'log.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )

    sys.stdout = logging.StreamHandler(sys.stdout)

    # merge original parameters and additional parameters
    params = {**params, **additional_params}

    # convert lists to ndarray
    params = {key: np.array(value) if isinstance(value, list) else value for key, value in params.items()}

    # dump params file
    with open(os.path.join(OUTPUT_DIRECTORY, 'params.json'), 'w') as f:
        # convert ndarray to lists
        params_json = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in params.items()}
        json.dump(params_json, f, indent=4)

    # test_optimizers()
    test_market_impact()

    logging.info(f"Successfully finished after {time.time() - start_time:.4f} seconds.")
