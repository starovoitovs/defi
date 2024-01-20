import json
import logging
import os
import sys
from datetime import datetime
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from defi.algorithms import gradient_descent, cvx
from defi.returns import generate_returns
from defi.utils import get_var_cvar, plot_metric
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
    results_metrics = np.full((len(algorithms), N_metrics), np.nan)

    N_completed = 0

    for j, algorithm in enumerate(algorithms):

        try:

            func, kwargs = algorithm['func'], algorithm['kwargs']

            portfolio_weights, *_ = func(returns, params, **kwargs)

            # need this, because zero submission into pools is not allowed
            eps = 1e-8
            portfolio_weights = eps + portfolio_weights
            portfolio_weights /= np.sum(portfolio_weights)

            # compute returns and metrics
            portfolio_returns = returns @ portfolio_weights
            var, cvar = get_var_cvar(portfolio_returns, params['alpha'])
            q_out = np.mean(portfolio_returns >= params['zeta'])

            results_weights[j, :] = portfolio_weights
            results_metrics[j, 0] = cvar
            results_metrics[j, 1] = q_out

            N_completed += 1

        except Exception as e:
            logging.info(f"Algorithm {j} failed: {e}")
            continue

    return results_weights, results_metrics, N_completed


def iterate(params, diffs, update_returns=False):
    # always start with uniformly distributed portfolio
    initial_weights = np.repeat(1., params['N_pools']) / params['N_pools']
    returns = generate_returns(params, initial_weights)

    N_algorithms = 4
    N_metrics = 2

    all_weights = np.full((len(diffs), N_algorithms, params['N_pools']), np.nan)
    all_metrics = np.full((len(diffs), N_algorithms, N_metrics), np.nan)

    for i, diff in enumerate(diffs):
        results_weights, results_metrics, N_completed = optimize(returns, {**params, **diff})
        all_weights[i, :, :] = results_weights
        all_metrics[i, :, :] = results_metrics

        # if failed for all constraints apart from unconstrained, early stoppage
        if N_completed <= 1:
            break

        # if update_returns=True, generate new returns with new weights for the next iteration
        if update_returns:

            # select weights where the algorithms has finished, constraint is satisfied and CVaR is best
            arr = np.where(results_metrics[:, 1] >= params['q'], results_metrics[:, 0], -np.inf)

            # if no algorithm satisfies the constraint, break
            if np.all(arr == -np.inf):
                break

            idx = np.argmax(arr)
            best_weights = results_weights[idx]
            returns = generate_returns(params, best_weights)

    return all_weights, all_metrics


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

    # # optimization test for various values of q
    # qs = np.linspace(0.51, 0.71, 11)
    # all_weights, all_metrics = iterate(params, [{'q': q} for q in qs])
    #
    # # plot
    # fig, ax = plt.subplots(ncols=3, figsize=(18, 4))
    # plot_metric(ax[0], qs, all_weights[:, :, 0], label='q', title='Weight of asset0')
    # plot_metric(ax[1], qs, all_metrics[:, :, 0], label='q', title='Portfolio CVaR')
    # plot_metric(ax[2], qs, all_metrics[:, :, 1], label='q', title='$P(r \geq \zeta)$')
    #
    # # plot an extra line to visualize the constraint
    # xmin, xmax = ax[2].get_xlim()
    # qs_plot = np.linspace(xmin, xmax, 101)
    # ax[2].plot(qs_plot, qs_plot, ls='--')
    #
    # fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'optimization_metrics.pdf'))

    # market impact iteration
    iterations = np.arange(params['N_iterations_mi'])
    all_weights, all_metrics = iterate(params, [{} for _ in iterations], update_returns=True)

    # plot
    fig, ax = plt.subplots(ncols=3, figsize=(18, 4))
    plot_metric(ax[0], iterations, all_weights[:, :, 0], label='N_iterations', title='Weight of asset0')
    plot_metric(ax[1], iterations, all_metrics[:, :, 0], label='N_iterations', title='Portfolio CVaR')
    plot_metric(ax[2], iterations, all_metrics[:, :, 1], label='N_iterations', title='$P(r \geq \zeta)$')

    # plot an extra line to visualize the constraint
    xmin, xmax = ax[2].get_xlim()
    ax[2].hlines(params['q'], xmin, xmax, color='k', linestyles='--')

    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'optimization_metrics.pdf'))
