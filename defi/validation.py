import logging
import os
import time

import numpy as np
from defi.optimization import optimize_iterate


def test_optimize(params):
    """
    Perform market impact iteration.
    :param params: Initial parameters
    :return:
    """
    start_time = time.time()
    logging.info(f"Starting `test_optimize`.")

    all_weights, all_returns, all_metrics = optimize_iterate(params, params['N_iterations_refinement'])

    os.makedirs(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy'), exist_ok=True)
    np.save(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy', 'weights.npy'), all_weights)
    np.save(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy', 'returns.npy'), all_returns)
    np.save(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'numpy', 'metrics.npy'), all_metrics)

    logging.info("Weights:")
    logging.info(all_weights)
    logging.info("Metrics (CVaR algorithm, CVaR actual, Empirical CDF):")
    logging.info(all_metrics)

    end_time = time.time() - start_time
    logging.info(f"Successfully finished `test_optimize` after {end_time:.3f} seconds.")
