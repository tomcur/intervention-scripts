#!/usr/bin/env bash

#SBATCH --partition=gpu
#SBATCH --job-name=int-train-imitation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=54:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:tesla:1

. ./prepare-intervention-env.sh

intervention-learning train-imitation \
    -d /gpfs/hpc/projects/Bolt/end-to-end/carla/datasets/2020-10-23T17:05:39+0300-teacher-examples \
    -o /gpfs/hpc/projects/Bolt/end-to-end/carla/checkpoints/200-10-26-imitation-2 \
    --num-epochs 15
