import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_mi_convergence(results_weights, results_metrics, params):
    fig, ax = plt.subplots(figsize=(12, 4), ncols=2)

    ax[0].plot(results_metrics, marker='o')
    ax[0].set_title(f"CVaR; x0 = {params['x_0']}; Rx0 = {params['Rx0']}")

    ax[1].pie(results_weights, labels=range(params['N_pools']))
    ax[1].set_title('Portfolio allocation')


def plot_hist(log_ret):
    zeta = 0.05
    alpha = 0.95

    sns.kdeplot(log_ret)

    p = np.mean(log_ret >= zeta)
    qtl = np.quantile(-log_ret, alpha)
    cvar = np.mean(-log_ret[-log_ret >= qtl])

    plt.axvline(-qtl, c='y', linestyle='--', label=f'VaR={-qtl:4.2}')
    plt.axvline(-cvar, c='r', linestyle='--', label=f'CVaR={-cvar:4.2}')
    plt.axvline(zeta, c='g', linestyle='--', label=f'P={p}')
    plt.legend()
    plt.show()


def plot_2d(returns):
    n_assets = returns.shape[1]

    fig, ax = plt.subplots(nrows=n_assets, ncols=n_assets, figsize=(2 * n_assets + 2, 2 * n_assets), constrained_layout=True)
    fig.suptitle('2D marginals', fontsize=20)

    for i in range(n_assets):
        for j in range(n_assets):
            if i < j:
                ax[i][j].hist2d(returns[:, i], returns[:, j], bins=50, range=[[-0.3, 0.3], [-0.3, 0.3]], cmap='viridis')
            if i > j:
                fig.delaxes(ax[i][j])
        sns.kdeplot(data=returns[:, i], ax=ax[i][i], fill=True, label='train')
        ax[i][i].legend()

    plt.show()


def plot_metric(ax, xs, metrics, line_labels, xlabel=None, title=None):

    for i, label in enumerate(line_labels):
        ax.plot(xs, metrics[:, i], marker='o', label=label)

    ax.legend()

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)


def get_var_cvar_empcdf(returns, alpha, zeta):
    var = np.quantile(returns, alpha, axis=0)
    cvar = np.sum(returns * (returns <= var), axis=0) / np.sum(returns <= var, axis=0) / (1 - alpha)
    empcdf = np.where(~np.any(np.isnan(returns), axis=0), np.mean(returns >= zeta, axis=0), np.nan)
    return var, cvar, empcdf
