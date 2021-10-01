#!/usr/bin/env bash

EPISODES_PER_JOB=5
JOBS=40
DATA_DIRECTORY=/gpfs/hpc/projects/Bolt/end-to-end/carla/datasets/$(date --iso-8601=seconds)-teacher-examples

job=1
while [ $job -le $JOBS ]; do
    NUMBER_OF_EPISODES=$EPISODES_PER_JOB \
    INTERVENTION_DATASET_DIRECTORY=$DATA_DIRECTORY \
        sbatch ./job-collect-teacher-examples.sh
    job=$(($job + 1))
done
