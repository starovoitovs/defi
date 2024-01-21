import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import norm
import torch


def cvx(returns, params, mode, **kwargs):
    assert params['q'] > 0.5

    weights = cp.Variable((params['N_pools'],))
    X = returns @ weights

    Z = cp.Variable((params['batch_size'],))
    var = cp.Variable((1,))
    cvar = var + 1. / (params['alpha'] * params['batch_size']) * cp.sum(Z)

    # generic weight constraints
    constraints = [cp.sum(weights) == 1., weights <= 1., weights * 1000 >= 0]

    # CVaR constraints
    constraints += [Z * 1000 >= 0, Z * 1000 >= -X * 1000 - var * 1000]

    # unconstrained
    if mode == 0:
        pass
    # lower bound: average of emp cdf: - might not be a valid constraint! unless VaR subadditive
    elif mode == 1:
        emp_cdf = np.mean(returns >= params['zeta'], axis=0)
        constraints += [emp_cdf @ weights * 1000 >= params['q'] * 1000]
    # normal approximation with SOC constraint
    elif mode == 2:
        mean, cov = np.mean(returns, axis=0), np.cov(returns.T)
        sqrtcov = sqrtm(cov)
        constraints += [cp.SOC((-params['zeta'] + mean @ weights) / norm.ppf(params['q']), sqrtcov @ weights)]

    objective = cp.Minimize(cvar * 1000)

    # possible solvers: "ECOS", "SCS", "OSQP", "CVXOPT"
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver="SCS")

    return weights.value,


def gradient_descent(returns, params, **kwargs):
    dtype = torch.float
    device = torch.device("cpu")

    returns_t = torch.tensor(returns, device=device, dtype=dtype)
    weights_t = torch.tensor(params['weights'], device=device, dtype=dtype, requires_grad=True)

    all_losses = torch.empty((0, 4))
    all_weights = torch.empty((0, params['N_pools']))

    for t in range(params['N_iterations_gd']):
        portfolio_returns_t = returns_t @ weights_t / torch.sum(weights_t)

        var = torch.quantile(portfolio_returns_t, params['alpha'], axis=0)
        cvar = torch.sum(portfolio_returns_t * (portfolio_returns_t <= var), axis=0) / torch.sum(portfolio_returns_t <= var, axis=0) / (1 - params['alpha'])

        # chance constraint
        # loss1 = (torch.mean((portfolio_returns >= zeta).float()) - q) ** 2
        loss0 = torch.relu(params['q'] - torch.mean(torch.sigmoid(1000 * (portfolio_returns_t - params['zeta'])))) ** 2

        # ensure that weights not negative
        loss1 = torch.mean(torch.relu(-weights_t))

        # ensure that weights sum to one
        loss2 = torch.square(torch.sum(weights_t) - 1.)

        # CVaR constraints
        loss3 = -cvar

        # total loss
        loss = params['loss_weights_gd'][0] * loss0 + params['loss_weights_gd'][1] * loss1 + params['loss_weights_gd'][2] * loss2 + params['loss_weights_gd'][3] * loss3

        all_losses = torch.cat([all_losses, torch.tensor([loss0, loss1, loss2, loss3]).reshape(1, -1)])
        all_weights = torch.cat([all_weights, weights_t.reshape(1, -1)])

        loss.backward()
        with torch.no_grad():
            weights_t -= params['learning_rate_gd'] * weights_t.grad
            weights_t.grad = None

    weights_n = weights_t.detach().numpy()
    weights_n = np.maximum(weights_n, 0)
    weights_n /= np.sum(weights_n)

    return weights_n, all_losses.detach().numpy()
