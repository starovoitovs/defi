import json
import os
import re

import numpy as np
from matplotlib import pyplot as plt

from defi.optimization import iterate
from defi.returns import generate_returns
from defi.utils import get_var_cvar_empcdf, plot_metric


def test_returns(params, params_diffs, xlabels, xaxes):
    from main import CURRENT_TIMESTAMP, OUTPUT_DIRECTORY

    # start with uniformly distributed portfolio, unless overridden
    params['weights'] = np.repeat(1., params['N_pools']) / params['N_pools']

    # plot
    fig, ax = plt.subplots(nrows=len(params_diffs), ncols=4, figsize=(24, max(5, 4 * len(params_diffs))), gridspec_kw={'hspace': 0.5}, squeeze=False)
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle(f"TEST RETURNS – {CURRENT_TIMESTAMP}")

    for idx, (diffs, xlabel, xaxis) in enumerate(zip(params_diffs, xlabels, xaxes)):

        # remove axis from the plot in the first column
        ax[idx][0].axis('off')

        returns = np.full((len(diffs), params['batch_size'], params['N_pools']), np.nan)
        weights = np.full((len(diffs), params['N_pools']), np.nan)

        for i, diff in enumerate(diffs):
            returns[i, :, :] = generate_returns({**params, **diff})
            weights[i, :] = {**params, **diff}['weights']

        portfolio_returns = np.einsum('ijk,ik->ji', returns, weights)

        _, portfolio_cvar, portfolio_empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])
        _, marginal_cvar, marginal_empcdf = get_var_cvar_empcdf(np.transpose(returns, axes=(1, 0, 2)), params['alpha'], params['zeta'])

        line_labels = [f"pool{i}" for i in range(params['N_pools'])]
        plot_metric(ax[idx][1], xaxis, marginal_empcdf, xlabel=xlabel, title='$P(r \geq \zeta)$')
        plot_metric(ax[idx][1], xaxis, portfolio_empcdf.reshape(-1, 1), color='k', ls='dashed')
        plot_metric(ax[idx][2], xaxis, marginal_cvar, xlabel=xlabel, title='CVaR')
        plot_metric(ax[idx][2], xaxis, portfolio_cvar.reshape(-1, 1), color='k', ls='dashed')
        plot_metric(ax[idx][3], xaxis, weights, line_labels=line_labels, xlabel=xlabel, title='Pool weights')

        if re.match("^w(\\d+)$", xlabel):
            for i in range(1, 4):
                ymin, ymax = ax[idx][i].get_ylim()
                ax[idx][i].vlines(1. / params['N_pools'], ymin, ymax, color='#00000099', linestyles='dotted')

    # output params in text on the plot
    params_text = {key: f"[{', '.join(['{num:.3f}'.format(num=x) for x in value])}]" if isinstance(value, list) or isinstance(value, np.ndarray) else value for key, value in params.items()}
    ax[0][0].text(0, ax[0][0].get_ylim()[1], json.dumps(params_text, indent=4), ha='left', va='top')

    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'test_returns.pdf'))


def test_optimizers(params, diffs, xlabel, xaxes):
    from main import CURRENT_TIMESTAMP, OUTPUT_DIRECTORY

    # start with uniformly distributed portfolio, unless overridden
    params['weights'] = np.repeat(1., params['N_pools']) / params['N_pools']

    all_weights, all_best_weights, all_returns = iterate(params, diffs,
                                                         break_on_constraint_violation=False,
                                                         store_initial_weights_and_returns=False,
                                                         update_returns=False)

    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_weights.npy'), all_weights)
    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_best_weights.npy'), all_best_weights)
    np.save(os.path.join(OUTPUT_DIRECTORY, 'all_returns.npy'), all_returns)

    # obtain portfolio returns and metrics: (N_return, N_diff, N_algorithm)
    portfolio_returns = np.einsum('jik,ilk->jil', all_returns, all_weights)
    _, portfolio_cvar, portfolio_empcdf = get_var_cvar_empcdf(portfolio_returns, params['alpha'], params['zeta'])

    # obtain marginal returns and metrics
    initial_returns = all_returns[:, 0, :]  # when testing optimizers, returns remain the same, so can take e.g. 0th ones
    _, marginal_cvar, marginal_empcdf = get_var_cvar_empcdf(initial_returns, params['alpha'], params['zeta'])

    # plot
    fig, ax = plt.subplots(ncols=5, figsize=(30, 5), gridspec_kw={'hspace': 0.5})
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle(f"TEST OPTIMIZERS – {CURRENT_TIMESTAMP}")

    # output params in text on the plot
    params_text = {key: f"[{', '.join(['{num:.3f}'.format(num=x) for x in value])}]" if isinstance(value, list) or isinstance(value, np.ndarray) else value for key, value in params.items()}
    ax[0].text(0, ax[0].get_ylim()[1], json.dumps(params_text, indent=4), ha='left', va='top')
    ax[0].axis('off')

    # output returns characteristic as barplots
    bar_width = 0.15
    pool_range = np.arange(initial_returns.shape[1])
    ax[1].bar(pool_range, np.mean(initial_returns, axis=0), width=bar_width, label='Mean')
    ax[1].bar(bar_width + pool_range, np.std(initial_returns, axis=0), width=bar_width, label='Std')
    ax[1].bar(2 * bar_width + pool_range, marginal_cvar, width=bar_width, label='Marginal CVaR')
    ax[1].bar(3 * bar_width + pool_range, marginal_empcdf, width=bar_width, label='Marginal $P(r \geq \zeta)$')
    ax[1].legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax[1].set_xlabel('Pool')
    ax[1].set_title("Properties of initial returns")

    line_labels = 'unconstrained', 'average cdf constraint', 'gaussian constraint', 'backprop'
    plot_metric(ax[2], xaxes, all_weights[:, :, 0], xlabel=xlabel, title='Weight of pool0')
    plot_metric(ax[3], xaxes, portfolio_cvar, xlabel=xlabel, title='Portfolio CVaR')
    plot_metric(ax[4], xaxes, portfolio_empcdf, line_labels, xlabel=xlabel, title='$P(r \geq \zeta)$')

    # plot an extra line to visualize the constraint, if testing against q
    if xlabel == 'q':
        xmin, xmax = ax[4].get_xlim()
        qs_plot = np.linspace(xmin, xmax, 101)
        ax[4].plot(qs_plot, qs_plot, color='#00000099', ls='dotted')

    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'test_optimization.pdf'))


def test_market_impact(params):
    from main import CURRENT_TIMESTAMP, OUTPUT_DIRECTORY

    iterations = np.arange(params['N_iterations_mi'])

    # start with uniformly distributed portfolio, unless overridden
    params['weights'] = np.repeat(1., params['N_pools']) / params['N_pools']

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
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(30, 8), gridspec_kw={'hspace': 0.5, 'width_ratios': [1.2, 1, 1, 1, 1]})
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle(f"MARKET IMPACT – {CURRENT_TIMESTAMP}")

    line_labels = 'unconstrained', 'average cdf constraint', 'gaussian constraint', 'backprop'
    plot_metric(ax[0][1], iterations, all_weights[:, :, 0], xlabel='N_iterations', title='Weight of pool0')
    plot_metric(ax[0][2], iterations, portfolio_cvar, xlabel='N_iterations', title='Portfolio CVaR')
    plot_metric(ax[0][3], iterations, portfolio_empcdf, xlabel='N_iterations', title='Portfolio $P(r \geq \zeta)$')

    # plot an extra line to visualize the constraint q
    xmin, xmax = ax[0][3].get_xlim()
    ax[0][3].hlines(params['q'], xmin, xmax, color='#00000099', linestyles='dotted')

    plot_metric(ax[0][4], iterations, np.mean(portfolio_returns, axis=0), line_labels=line_labels, xlabel='N_iterations', title='Portfolio mean')

    # plot return characteristics as function of iteration
    line_labels = [f"pool{i}" for i in range(params['N_pools'])]
    plot_metric(ax[1][1], iterations, all_best_weights, xlabel='N_iterations', title='Best weights')
    plot_metric(ax[1][2], iterations, marginal_cvar, xlabel='N_iterations', title='Marginal CVaR')
    plot_metric(ax[1][3], iterations, marginal_empcdf, xlabel='N_iterations', title='Marginal $P(r_i \geq \zeta)$')

    # plot an extra line to visualize the constraint q
    xmin, xmax = ax[1][3].get_xlim()
    ax[1][3].hlines(params['q'], xmin, xmax, color='#00000099', linestyles='dotted')

    plot_metric(ax[1][4], iterations, np.mean(all_returns, axis=0), line_labels=line_labels, xlabel='N_iterations', title='Marginal mean')

    # merge two axes on the left
    ax_text_left = fig.add_subplot(111, frame_on=False)
    ax[0][0].axis('off')
    ax[1][0].axis('off')

    # output params in text on the plot
    params_text = {key: f"[{', '.join(['{num:.3f}'.format(num=x) for x in value])}]" if isinstance(value, list) or isinstance(value, np.ndarray) else value for key, value in params.items()}
    ax_text_left.text(0, ax_text_left.get_ylim()[1], json.dumps(params_text, indent=4), ha='left', va='top')
    ax_text_left.axis('off')

    fig.savefig(os.path.join(OUTPUT_DIRECTORY, 'test_market_impact.pdf'))

