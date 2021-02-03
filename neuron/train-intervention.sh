#!/usr/bin/env bash

echo >&2 "Running on: $(hostname)"
echo >&2 "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export INTERVENTION_NEGATIVE_LEARNING_DECAY_INITIAL="0.0"
export INTERVENTION_NEGATIVE_LEARNING_DECAY_TIME="10.0"

. ./prepare-intervention-env.sh

LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    intervention-learning train-intervention \
    --intervention-dataset-directory ~/datasets/2021-01-16T01:38:14+0200-on-policy \
    --imitation-dataset-directory ~/datasets/2020-12-07T22:06:19+0200-teacher-examples \
    --initial-checkpoint ~/checkpoints/2020-12-18-imitation-ce/24.pth \
    -o ~/checkpoints/"2021-01-25-intervention-ce-di${INTERVENTION_NEGATIVE_LEARNING_DECAY_INITIAL}-dt${INTERVENTION_NEGATIVE_LEARNING_DECAY_TIME}" \
    --num-epochs 35
