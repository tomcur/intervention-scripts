#!/usr/bin/env bash

echo >&2 "Running on: $(hostname)"
echo >&2 "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

. ./prepare-intervention-env.sh

target=teacher-prediction
loss=cross-entropy

case "${target}" in
    teacher-prediction) target_str=t ;;
    location) target_str=l ;;
esac

case "${loss}" in
    cross-entropy) loss_str=ce ;;
    expected-value) loss_str=ev ;;
esac

# CHECKPOINT=$(ls ~/checkpoints/2021-06-09-imitation-${target_str}-${loss_str}/* | sort -V | tail -n 1)
# echo "resuming from ${CHECKPOINT}"

LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    intervention-learning train-imitation \
    --dataset-directory /data/end-to-end/datasets/20210608T135147-teacher \
    -o ~/checkpoints/"2021-08-11-imitation-nobatchnorm-${target_str}-${loss_str}" \
    --target-source ${target} \
    --loss-type ${loss} \
    --num-epochs 50
    # --initial-checkpoint "${CHECKPOINT}"
