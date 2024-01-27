import numpy as np
import torch


def gradient_descent(xs, ys, rxs, rys, phis, params, **kwargs):

    dtype = torch.float
    device = torch.device("cpu")

    xs_t = torch.tensor(xs, device=device, dtype=dtype)
    ys_t = torch.tensor(ys, device=device, dtype=dtype)
    rxs_t = torch.tensor(rxs, device=device, dtype=dtype)
    rys_t = torch.tensor(rys, device=device, dtype=dtype)
    phi_t = torch.tensor(phis, device=device, dtype=dtype)

    w0_t = torch.tensor(params['weights'], device=device, dtype=dtype)
    weights_t = torch.tensor(params['weights'], device=device, dtype=dtype, requires_grad=True)

    all_weights = torch.empty((0, params['N_pools']))

    log_x0 = np.log(params['x_0'])

    for t in range(params['N_iterations_gd']):

        x_burn_t = xs_t @ (weights_t / w0_t) / torch.sum(weights_t)
        y_burn_t = ys_t @ (weights_t / w0_t) / torch.sum(weights_t)
        x_swap_t = y_burn_t * (1 - phi_t) * rxs_t / (rys_t + (1 - phi_t) * y_burn_t)
        portfolio_returns_t = torch.log(x_burn_t + x_swap_t) - log_x0

        qtl = torch.quantile(-portfolio_returns_t, params['alpha'])
        cvar = torch.mean(-portfolio_returns_t[-portfolio_returns_t >= qtl])

        # chance constraint
        loss0 = torch.relu(params['q'] - torch.mean(torch.sigmoid(1000 * (portfolio_returns_t - params['zeta'])))) ** 2

        # ensure that weights not negative
        loss1 = torch.mean(torch.relu(-weights_t))

        # ensure that weights sum to one
        loss2 = torch.square(torch.sum(weights_t) - 1.)

        # CVaR constraints
        loss3 = cvar

        # total loss
        loss = params['loss_weights_gd'][0] * loss0 + params['loss_weights_gd'][1] * loss1 + params['loss_weights_gd'][2] * loss2 + params['loss_weights_gd'][3] * loss3

        all_weights = torch.cat([all_weights, weights_t.reshape(1, -1)])

        loss.backward()
        with torch.no_grad():
            weights_t -= params['learning_rate_gd'] * weights_t.grad
            weights_t.grad = None

    weights_n = weights_t.detach().numpy()
    weights_n = np.maximum(weights_n, 0)
    weights_n /= np.sum(weights_n)

    return weights_n, cvar.detach().numpy()
