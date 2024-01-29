# Run optimizer

You can run an experiment as follows:

    EXPERIMENT_NAME=[experiment_name] WEIGHTS=0.1,0.1,0.1,0.1,0.1,0.5 ./optimize.sh

It will create a directory `_output/[experiment_name]` with outputs. You can specify initial weights as above (need to be comma-separated floats without SPACE). This will run the experiment with 50 different values of the seed with `batch_size=1000`.
