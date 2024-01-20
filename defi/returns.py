import numpy as np

from defi.amm import amm


def generate_returns(params, weights):
    np.random.seed(params['seed'])

    pools = amm(Rx=params['Rx0'], Ry=params['Ry0'], phi=params['phi'])

    xs_0 = weights * params['x_0']
    l = pools.swap_and_mint(xs_0)

    end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = pools.simulate(params['kappa'], params['p'], params['sigma'], T=params['T'], batch_size=params['batch_size'])

    x_T = np.array([pool.burn_and_swap(l) for pool in end_pools])
    returns = np.log(x_T) - np.log(xs_0)

    return returns


def generate_gaussian_returns(params):
    # 2 pools
    np.random.seed(params['seed'])
    returns = np.array((0.07, 0.05)) + np.random.standard_normal((1000, params['N_pools'])) * np.array((0.05, 0.03))[None, :]
    return returns
