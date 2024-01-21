import json
import logging
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from defi.algorithms import gradient_descent, cvx
from defi.returns import generate_returns
from defi.utils import get_var_cvar_empcdf, plot_metric
from params import params

sns.set_theme(style="ticks")

additional_params = {
    # learning rate of the gradient descnet
    'learning_rate_gd': 1e-2,
    # number of iterations for gradient descent
    'N_iterations_gd': 2000,
    # loss weights in gradient descent
    'loss_weights': [1e3, 1., 1., 1.],
    # number of iterations in market impact model
    'N_iterations_mi': 10,
}


def optimize(returns, params):
    algorithms = []

    # unconstrained, for demostration purposes
    algorithms += [{'func': cvx, 'kwargs': {'mode': 0}}]

    # lower bound: average of emp cdf: - might not be a valid constraint! unless VaR subadditive
    algorithms += [{'func': cvx, 'kwargs': {'mode': 1}}]

    # normal approximation with SOC constraint
    algorithms += [{'func': cvx, 'kwargs': {'mode': 2}}]

    # gradient descent with backprop
    algorithms += [{'func': gradient_descent, 'kwargs': {}}]

    N_metrics = 2

    results_weights = np.full((len(algorithms), params['N_pools']), np.nan)

    N_completed = 0

    for j, algorithm in enumerate(algorithms):

        try:

            func, kwargs = algorithm['func'], algorithm['kwargs']

            portfolio_weights, *_ = func(returns, params, **kwargs)

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


def iterate(params, diffs, break_on_constraint_violation=False, update_returns=False):
    # always start with uniformly distributed portfolio
    initial_weights = np.repeat(1., params['N_pools']) / params['N_pools']
    returns = generate_returns(params, initial_weights)

    N_algorithms = 4

    all_returns = np.full((params['batch_size'], len(diffs), params['N_pools']), np.nan)
    all_weights = np.full((len(diffs), N_algorithms, params['N_pools']), np.nan)
    all_best_weights = np.full((len(diffs), params['N_pools']), np.nan)

    for i, diff in enumerate(diffs):

        all_returns[:, i, :] = returns

        results_weights, N_completed = optimize(returns, {**params, **diff})
        all_weights[i, :, :] = results_weights

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
        best_weights = results_weights[idx]
        all_best_weights[i, :] = best_weights

        # if update_returns=True, generate new returns with new weights for the next iteration
        if update_returns:
            returns = generate_returns(params, best_weights)

    return all_weights, all_best_weights, all_returns


if __name__ == '__main__':
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

    # optimization test for various values of q
    qs = np.linspace(0.51, 0.71, 11)
    all_weights, all_best_weights, all_returns = iterate(params, [{'q': q} for q in qs], break_on_constraint_violation=False, update_returns=False)

    # compute returns and metrics: (N_return, N_diff, N_algorithm)
    portfolio_returns = np.einsum('jik,ilk->jil', all_returns, all_weights)
    var, cvar, empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])

    # plot
    fig, ax = plt.subplots(ncols=3, figsize=(18, 4))
    line_labels = 'unconstrained', 'average cdf constraint', 'gaussian constraint', 'backprop'
    plot_metric(ax[0], qs, all_weights[:, :, 0], line_labels, xlabel='q', title='Weight of asset0')
    plot_metric(ax[1], qs, cvar, line_labels, xlabel='q', title='Portfolio CVaR')
    plot_metric(ax[2], qs, empcdf, line_labels, xlabel='q', title='$P(r \geq \zeta)$')

    # plot an extra line to visualize the constraint
    xmin, xmax = ax[2].get_xlim()
    qs_plot = np.linspace(xmin, xmax, 101)
    ax[2].plot(qs_plot, qs_plot, ls='--')

    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'optimization_metrics.pdf'))

    # # market impact iteration
    # iterations = np.arange(params['N_iterations_mi'])
    # all_weights, all_best_weights, all_returns = iterate(params, [{} for _ in iterations], break_on_constraint_violation=True, update_returns=True)
    #
    # # store the outputs
    # pd.DataFrame(all_weights).to_csv(os.path.join('all_weights.csv'), header=False, index=False)
    # pd.DataFrame(all_best_weights).to_csv(os.path.join('all_best_weights.csv'), header=False, index=False)
    # pd.DataFrame(all_returns).to_csv(os.path.join('all_returns.csv'), header=False, index=False)
    #
    # # compute returns and metrics: (N_return, N_diff, N_algorithm)
    # portfolio_returns = np.einsum('jik,ilk->jil', all_returns, all_weights)
    # _, portfolio_cvar, portfolio_empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])
    #
    # # plots
    # fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 8))
    # line_labels = 'unconstrained', 'average cdf constraint', 'gaussian constraint', 'backprop'
    # plot_metric(ax[0][0], iterations, all_weights[:, :, 0], line_labels, xlabel='N_iterations', title='Weight of asset0')
    # plot_metric(ax[0][1], iterations, portfolio_cvar, line_labels, xlabel='N_iterations', title='Portfolio CVaR')
    # plot_metric(ax[0][2], iterations, portfolio_empcdf, line_labels, xlabel='N_iterations', title='$P(r \geq \zeta)$')
    #
    # # plot an extra line to visualize the constraint q
    # xmin, xmax = ax[0][2].get_xlim()
    # ax[0][2].hlines(params['q'], xmin, xmax, color='k', linestyles='--')
    #
    # # plot return characteristics as function of iteration
    # line_labels = [f"asset{i}" for i in range(params['N_pools'])]
    # _, marginal_cvar, marginal_empcdf = get_var_cvar_empcdf(all_returns, params['alpha'], params['zeta'])
    # plot_metric(ax[1][0], iterations, all_best_weights, line_labels, xlabel='N_iterations', title='Best weights')
    # plot_metric(ax[1][1], iterations, np.mean(all_returns, axis=0), line_labels, xlabel='N_iterations', title='Marginal Mean')
    # plot_metric(ax[1][2], iterations, marginal_cvar, line_labels, xlabel='N_iterations', title='Marginal CVaR')
    # plot_metric(ax[1][3], iterations, marginal_empcdf, line_labels, xlabel='N_iterations', title='Marginal $P(r_i \geq \zeta)$')
    #
    # # plot an extra line to visualize the constraint q
    # xmin, xmax = ax[1][3].get_xlim()
    # ax[1][3].hlines(params['q'], xmin, xmax, color='k', linestyles='--')
    #
    # fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'optimization_metrics.pdf'))
