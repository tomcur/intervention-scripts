#!/usr/bin/env bash

. ~/.bashrc
conda activate intervention

export XDG_RUNTIME_DIR=/tmp
export INTERVENTION_DATASET_DIRECTORY=/gpfs/hpc/projects/Bolt/end-to-end/carla/datasets/$(date --iso-8601=seconds)
export INTERVENTION_LBC_BIRDVIEW_CHECKPOINT=~/intervention-models/lbc-birdview/model-128.th
