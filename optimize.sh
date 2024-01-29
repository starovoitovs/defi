#!/bin/bash

for ((seed=4294967143; seed<=4294967192; seed++)); do
    SEED=$seed python main.py 
    done

    # Wait for all background processes to finish
    wait

