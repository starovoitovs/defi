#!/bin/bash

for ((seed=4294967143; seed<4294967143+$N_SEEDS; seed++)); do
    SEED=$seed python main.py &
    done

    # Wait for all background processes to finish
    wait

