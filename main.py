import json
import logging
import os
import sys
from datetime import datetime
import numpy as np
import seaborn as sns

from defi.optimization import optimize
from params import params, additional_params

sns.set_theme(style="ticks")


if __name__ == '__main__':

    exp_name = os.environ['EXPERIMENT_NAME'] if 'EXPERIMENT_NAME' in os.environ else 'misc'

    # override seed if provided
    if 'SEED' in os.environ:
        params['seed'] = int(os.environ['SEED'])
    
    # use uniform weights by default
    if 'WEIGHTS' in os.environ:
        additional_params['weights'] = np.array([float(weight) for weight in os.environ['WEIGHTS'].split(',')])

    CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIRECTORY = os.path.join(f"_output/{exp_name}", f"{CURRENT_TIMESTAMP}__{params['seed']}")

    os.environ['RUN_ID'] = CURRENT_TIMESTAMP
    os.environ['OUTPUT_DIRECTORY'] = OUTPUT_DIRECTORY

    os.makedirs(os.environ['OUTPUT_DIRECTORY'], exist_ok=True)

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'log.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )

    logging.info(f"Logging into directory {OUTPUT_DIRECTORY}.")

    # ensure lines are not breaking when printing numpy array
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(linewidth=np.inf)

    sys.stdout = logging.StreamHandler(sys.stdout)

    # merge original parameters and additional parameters
    params = {**params, **additional_params}

    # convert lists to ndarray
    params = {key: np.array(value) if isinstance(value, list) else value for key, value in params.items()}

    # dump params file
    with open(os.path.join(os.environ['OUTPUT_DIRECTORY'], 'params.json'), 'w') as f:
        # convert ndarray to lists
        params_json = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in params.items()}
        json.dump(params_json, f, indent=4)

    optimize(params)
