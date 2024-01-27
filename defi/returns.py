import numpy as np

from defi.amm import amm


def generate_market(params):
    np.random.seed(params['seed'])

    pools = amm(Rx=params['Rx0'], Ry=params['Ry0'], phi=params['phi'])

    xs_0 = params['weights'] * params['x_0']
    l = pools.swap_and_mint(xs_0)

    end_pools, Rx_t, Ry_t, v_t, event_type_t, event_direction_t = pools.simulate(params['kappa'], params['p'], params['sigma'], T=params['T'], batch_size=params['batch_size'])

    xs, ys, rxs, rys, phis = [], [], [], [], []

    for end_pool in end_pools:

        x_burn, y_burn = end_pool.burn(l)
        y_burn_sum = np.sum(y_burn)

        max_pool, max_quote = None, -np.inf

        for i in range(params['N_pools']):
            y_swap = np.zeros(params['N_pools'])
            y_swap[i] = y_burn_sum
            quote = np.max(end_pool.swap_y_to_x(y_swap, quote=True))

            if quote > max_quote:
                max_quote = quote
                max_pool = i

        xs += [x_burn]
        ys += [y_burn]

        rxs += [end_pool.Rx[max_pool]]
        rys += [end_pool.Ry[max_pool]]
        phis += [end_pool.phi[max_pool]]

    xs, ys, rxs, rys, phis = np.array(xs), np.array(ys), np.array(rxs), np.array(rys), np.array(phis)

    return xs, ys, rxs, rys, phis
