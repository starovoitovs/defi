This is the code for the [SIAM/FME Code Quest 2023](https://sites.google.com/view/siagfme-codequest2023/home).

# Parameters

* `params.py` - original parameters from the challenge.
* `main.py` - contains `additional_params` object with custom parameters.

# Run optimizer

You can run a single experiment as follows:

    EXPERIMENT_NAME=[experiment_name] WEIGHTS=[weights] python main.py

You can run several experiments in parallel as follows:

    N_SEEDS=[n_seeds] EXPERIMENT_NAME=[experiment_name] WEIGHTS=[0.1,0.1,0.1,0.1,0.1,0.5] ./optimize.sh

In both cases, the parameters correspond to:

* `experiment_name` - directory in `_output` where the results are going to be written. Defaults to `misc`. 
* `weights` - initial weights. Comma-separated non-negative floats without SPACE adding up to 1, for example `WEIGHTS=0.1,0.1,0.1,0.1,0.1,0.5`. Defaults to `1/N_pools` for each pool.
* `n_seeds` - number of seeds used in the parallel run. Will generate output from `seed=4294967143` to `seed=4294967143+[n_seeds]-1`. Required.

# Output

Each run within experiment is written in the directory `_output/[experiment_name]/[timestamp]_[seed]`. The structure of the output directory is:

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

# Plotting

The notebook `plotting.ipynb` contains scripts used to generate plots.
