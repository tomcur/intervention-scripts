#!/usr/bin/env bash

EPISODES_PER_JOB=5
JOBS=40
DATA_DIRECTORY=/gpfs/hpc/projects/Bolt/end-to-end/carla/datasets/$(date --iso-8601=seconds)-on-policy
STUDENT_CHECKPOINT=/gpfs/hpc/projects/Bolt/end-to-end/carla/checkpoints/2020-12-18-imitation-ce/24.pth

job=1
while [ $job -le $JOBS ]; do
    NUMBER_OF_EPISODES=$EPISODES_PER_JOB \
    INTERVENTION_DATASET_DIRECTORY=$DATA_DIRECTORY \
    INTERVENTION_STUDENT_CHECKPOINT=$STUDENT_CHECKPOINT \
        sbatch ./job-collect-on-policy.sh
    job=$(($job + 1))
done
