This is the code for the [SIAM/FME Code Quest 2023](https://sites.google.com/view/siagfme-codequest2023/home).

# Parameters

Located in `params.py`.

* `params` - original parameters from the challenge.
* `additional_params` - object with custom parameters.

# Tests

Unit tests for the first part of the challenge are located in `tests.py`. They can be run by

    python tests.py

# Run optimizer

You can run a single experiment as follows:

    SEED=[seed] EXPERIMENT_NAME=[experiment_name] WEIGHTS=[weights] python main.py

You can run several experiments in parallel as follows:

    N_SEEDS=[n_seeds] EXPERIMENT_NAME=[experiment_name] WEIGHTS=[weights] ./optimize.sh

In both cases, the parameters correspond to:

* `experiment_name` - directory in `_output` where the results are going to be written. Defaults to `misc`. 
* `weights` - initial weights. Comma-separated non-negative floats without SPACE adding up to 1, for example `WEIGHTS=0.1,0.1,0.1,0.1,0.1,0.5`. Defaults to `[0.001, 0.332, 0.332, 0.332, 0.001, 0.001]`.
* `seed` - seed used in a single run. Defaults to the value in `params.py`.
* `n_seeds` - number of seeds used in the parallel run. Will generate output from `seed=4294967143` to `seed=4294967143+[n_seeds]-1`. Required.

# Output

Output for each run within experiment is written into the directory `_output/[experiment_name]/[timestamp]__[seed]`. The structure of the output directory is as follows:

    my_experiment
    ├── 20240129_232736__4294967143
    │   ├── numpy
    │   │   ├── metrics.npy
    │   │   ├── returns.npy
    │   │   └── weights.npy
    │   ├── log.log
    │   ├── params.json
    │   └── weights.csv

* `numpy/` - contains metrics, weights and returns for all iterations.
* `log.log` - contains among other things training output.
* `params.json` - dump of the parameters used for the run.
* `weights.csv` - best weights.

The file `numpy/metrics.npy` contains an array with 3 columns, which correspond to:
* `CVaR_algorithm` - objective of the algorithm which is usually not the actual CVaR due to market impact after different allocation.
* `CVaR_actual` - actual CVaR for the given weights.
* `ecdf` - empirical CDF `P(r ≥ ζ)` for the given returns for `ζ=0.05`, which in the end is supposed to exceed `q=0.8`.

# Plotting

The notebook `plotting.ipynb` contains scripts used to generate plots.
