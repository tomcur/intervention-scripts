#!/usr/bin/env bash

EPISODES_PER_JOB=10
JOBS=40
DATA_DIRECTORY=/gpfs/hpc/projects/Bolt/end-to-end/carla/datasets/$(date --iso-8601=seconds)-teacher-examples

for j in {1..$JOBS}; do
    NUMBER_OF_EPISODES=$EPISODES_PER_JOB \
    INTERVENTION_DATASET_DIRECTORY=$DATA_DIRECTORY \
        sbatch ./job-collect-teacher-examples.sh
done
