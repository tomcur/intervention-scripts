#!/usr/bin/env bash

# Run Carla and a teacher model to collect teacher driving data.

#SBATCH --partition=gpu
#SBATCH --job-name=int-collect-teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=54:30:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=20G
#SBATCH --gres=gpu:tesla:2

NUMBER_OF_EPISODES=${NUMBER_OF_EPISODES:-4}

case "$1" in
    "carla")
        # Start Carla
        cd "${INTERVENTION_CARLA_DIRECTORY}"
        echo >&2 "Starting forked Carla process (probably on GPU ${CUDA_VISIBLE_DEVICES})"

        DISPLAY= \
            ./CarlaUE4.sh -opengl -carla-world-port=$CARLA_WORLD_PORT
        ;;
    "intervention")
        . ./prepare-intervention-env.sh

        echo >&2 "Using GPU ${CUDA_VISIBLE_DEVICES} as CUDA visible device"
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

        srun --job-name=simulator --ntasks=1 --mem=6G --gres=gpu:tesla:1 --exclusive \
            ./job-collect-teacher-examples.sh carla &
        sleep 10

        export DATA_DIRECTORY=$(mktemp --directory)

        srun --job-name=collector --ntasks=1 --mem=13G --gres=gpu:tesla:1 --exclusive \
            ./job-collect-teacher-examples.sh intervention

        INTERVENTION_DATASET_DIRECTORY=${INTERVENTION_DATASET_DIRECTORY:-/gpfs/hpc/projects/Bolt/end-to-end/carla/datasets/$(date --iso-8601=seconds)}

        echo >&2 "Data collection stopped"
        echo >&2 "Merging data from ${DATA_DIRECTORY} into ${INTERVENTION_DATASET_DIRECTORY}"
        ./merge-datasets.sh "${DATA_DIRECTORY}" "${INTERVENTION_DATASET_DIRECTORY}"
        echo >&2 "Data synced"
        ;;
esac
