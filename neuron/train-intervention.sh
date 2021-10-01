#!/usr/bin/env bash

echo >&2 "Running on: $(hostname)"
echo >&2 "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export INTERVENTION_NEGATIVE_LEARNING_DECAY_INITIAL="0.0"
export INTERVENTION_NEGATIVE_LEARNING_DECAY_TIME="10.0"

. ./prepare-intervention-env.sh

target=teacher-prediction
loss=cross-entropy-swapped

case "${target}" in
    teacher-prediction) target_str=t ;;
    location) target_str=l ;;
esac

case "${loss}" in
    cross-entropy) loss_str=ce ;;
    cross-entropy-swapped) loss_str=ces ;;
    expected-value) loss_str=ev ;;
esac

LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    intervention-learning train-intervention \
    --intervention-dataset-directory /data/end-to-end/datasets/20210814T214335-intervention-2021-08-13-intervention-nobn-t-ce-di0.0-dt1000000.0-12 \
    --imitation-dataset-directory /data/end-to-end/datasets/20210608T135147-teacher \
    --initial-checkpoint ~/checkpoints/"2021-08-16-intervention-nobn-t-ces-di0.25-dt1000000.0/15.pth" \
    -o ~/checkpoints/"2021-08-16-intervention-nobn-after-ces-0.25-${target_str}-${loss_str}-di${INTERVENTION_NEGATIVE_LEARNING_DECAY_INITIAL}-dt${INTERVENTION_NEGATIVE_LEARNING_DECAY_TIME}" \
    --target-source ${target} \
    --loss-type ${loss} \
    --num-epochs 10
