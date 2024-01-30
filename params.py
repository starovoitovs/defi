# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:05:41 2023

@author: sebja
"""
import numpy as np

# the actual parameters

N_pools = 6
params = {'N_pools': N_pools,
          'Rx0': 100 * np.ones(N_pools),
          'Ry0': 1000 * np.ones(N_pools),
          'phi': 0.03 * np.ones(N_pools),
          'x_0': 10.,
          'alpha': 0.9,
          'q': 0.8,
          'zeta': 0.05,
          'batch_size': 1000,
          'kappa': np.array([0.25, 0.5, 0.5, 0.45, 0.45, 0.4, 0.3]),
          'sigma': np.array([1., 0.3, 0.5, 1., 1.25, 2, 4]),
          'p': np.array([0.45, 0.45, 0.4, 0.38, 0.36, 0.34, 0.3]),
          'T': 60,
          'seed': 4294967143}


additional_params = {
    # number of iteratations for the "iteration" (refinement) loop
    'N_iterations_main': 20,
    # learning rate of the gradient descent
    'learning_rate_gd': 1e-2,
    # number of iterations for gradient descent
    'N_iterations_gd': 2000,
    # loss weights in gradient descent
    'loss_weights_gd': [1e1, 1., 1., 1.],
    # minimum weight to ensure that we never put 0 coins in a pool
    'weight_eps': 1e-4,
    # initial weights used for generation of returns
    # 'weights': np.repeat(1., params['N_pools']) / params['N_pools'],
    'weights': [0.001, 0.332, 0.332, 0.332, 0.002, 0.001], # good guess
    # 'weights': [0.001, 0.001, 0.001, 0.001, 0.001, 0.995], # bad guess
}
