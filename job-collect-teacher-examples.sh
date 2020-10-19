#!/usr/bin/env bash

#SBATCH --partition=gpu
#SBATCH --job-name=int-collect-teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=54:30:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --gres=gpu:tesla:2
#SBATCH --exclude=falcon[6]

NUMBER_OF_EPISODES=4

case "$1" in
    "carla")
        # Start Carla
        cd "${INTERVENTION_CARLA_DIRECTORY}"
        echo >&2 "Starting forked Carla process (probably on GPU ${CUDA_VISIBLE_DEVICES})"

        DISPLAY= \
            ./CarlaUE4.sh -opengl
        ;;
    "intervention")
        . ./prepare-intervention-env.sh

        echo >&2 "Using GPU ${CUDA_VISIBLE_DEVICES} as CUDA visible device"
        DATA_DIRECTORY="${INTERVENTION_DATASET_DIRECTORY}/$(date --iso-8601)-teacher-examples"
        echo >&2 "Starting intervention learning data collection. Data directory: ${DATA_DIRECTORY}"

        LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
            xvfb-run \
            intervention-learning collect-teacher-examples \
            -t "${INTERVENTION_LBC_BIRDVIEW_CHECKPOINT}" \
            -n $NUMBER_OF_EPISODES \
            -d "${DATA_DIRECTORY}"
        ;;
    *)
        . ./prepare-carla-env.sh

        srun --ntasks=1 --mem=6G --gres=gpu:tesla:1 --exclusive ./job-collect-teacher-examples.sh carla &
        sleep 10
        srun --ntasks=1 --mem=13G --gres=gpu:tesla:1 --exclusive ./job-collect-teacher-examples.sh intervention

        echo >&2 "Data collection stopped"
        ;;
esac
