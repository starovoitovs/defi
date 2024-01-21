# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:05:41 2023

@author: sebja
"""
import numpy as np

# N_pools = 6
# params = {'N_pools': N_pools,
#           'Rx0': 100 * np.ones(N_pools),
#           'Ry0': 1000 * np.ones(N_pools),
#           'phi': 0.03 * np.ones(N_pools),
#           'x_0': 10,
#           'alpha': 0.1,
#           'q': 0.75,
#           'zeta': 0.0,z
#           'batch_size': 1_000,
#           'kappa': [0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#           'sigma': [1, 0.3, 0.5, 1.5, 1.75, 2, 2.25],
#           'p': [0.35, 0.3, 0.34, 0.33, 0.32, 0.31, 0.3],
#           'T': 60,
#           'seed': 4294967143}

N_pools = 6
params = {'N_pools': N_pools,
          'Rx0': 100 * np.ones(N_pools),
          'Ry0': 1000 * np.ones(N_pools),
          'phi': 0.03 * np.ones(N_pools),
          'x_0': 600.,
          'alpha': 0.05,
          'q': 0.6,
          'zeta': 0.05,
          'batch_size': 1_000,
          'kappa': np.array([0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
          'sigma': np.array([1, 0.5, 0.5, 1.5, 1.75, 2, 2.25]),
          'p': np.array([0.32, 0.34, 0.34, 0.33, 0.32, 0.31, 0.32]),
          'T': 60,
          'seed': 4294967143}

# N_pools = 2
# params = {'N_pools': N_pools,
#           'Rx0': 100 * np.ones(N_pools),
#           'Ry0': 1000 * np.ones(N_pools),
#           'phi': np.array([0.03, 0.03]),
#           'x_0': 200.,
#           'alpha': 0.05,
#           'q': 0.6,
#           'zeta': 0.05,
#           'batch_size': 1_000,
#           'kappa': np.array([0.5, 0.5, 0.5]),
#           'sigma': np.array([1., 1., 1.]),
#           'p': np.array([0.35, 0.35, 0.35]),
#           'T': 60,
#           'seed': 4294967143}
