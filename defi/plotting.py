import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_mi_convergence(results_weights, results_metrics, params):
    fig, ax = plt.subplots(figsize=(12, 4), ncols=2)

    ax[0].plot(results_metrics, marker='o')
    ax[0].set_title(f"CVaR; x0 = {params['x_0']}; Rx0 = {params['Rx0']}")

    ax[1].pie(results_weights, labels=range(params['N_pools']))
    ax[1].set_title('Portfolio allocation')


def plot_hist(returns):
    zeta = 0.05
    alpha = 0.95

    sns.kdeplot(returns)

    p = np.mean(returns >= zeta)
    qtl = np.quantile(-returns, alpha)
    cvar = np.mean(-returns[-returns >= qtl])

    plt.axvline(-qtl, c='y', linestyle='--', label=f'VaR={-qtl:4.2}')
    plt.axvline(-cvar, c='r', linestyle='--', label=f'CVaR={-cvar:4.2}')
    plt.axvline(zeta, c='g', linestyle='--', label=f'P={p}')
    plt.legend()
    plt.show()


def plot_2d(returns):
    N_pools = returns.shape[1]

    fig, ax = plt.subplots(nrows=N_pools, ncols=N_pools, figsize=(2 * N_pools + 2, 2 * N_pools), constrained_layout=True)
    fig.suptitle('2D marginals', fontsize=20)

    for i in range(N_pools):
        for j in range(N_pools):
            if i < j:
                ax[i][j].hist2d(returns[:, i], returns[:, j], bins=50, range=[[-0.3, 0.3], [-0.3, 0.3]], cmap='viridis')
            if i > j:
                fig.delaxes(ax[i][j])
        sns.kdeplot(data=returns[:, i], ax=ax[i][i], fill=True, label='train')
        ax[i][i].legend()

    plt.show()


def plot_metric(ax, xs, metrics, line_labels=None, xlabel=None, title=None, **kwargs):
    for i in range(metrics.shape[1]):
        label = None if line_labels is None else line_labels[i]
        ax.plot(xs, metrics[:, i], marker='o', label=label, **kwargs)

    if line_labels is not None:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)
