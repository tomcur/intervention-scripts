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



# Carla seems to always take the first GPU in CUDA_VISIBLE_DEVICES (it uses OpenGL).
# Let's give the second one to our application for CUDA.
CARLA_GPU="$(cut -d',' -f1 <<<$CUDA_VISIBLE_DEVICES)"
INTERVENTION_GPU="$(cut -d',' -f2 <<<$CUDA_VISIBLE_DEVICES)"


case "$1" in
  "carla")
    # Start Carla
    cd "${INTERVENTION_CARLA_DIRECTORY}"
    >&2 echo "Starting forked Carla process (probably on GPU ${CARLA_GPU})"

    DISPLAY= \
        ./CarlaUE4.sh -opengl
     ;;
  "intervention")
     . ./prepare-intervention-env.sh

    >&2 echo "Using GPU ${INTERVENTION_GPU} as CUDA visible device"
    DATA_DIRECTORY="${INTERVENTION_DATASET_DIRECTORY}/$(date --iso-8601)-teacher-examples"
    >&2 echo "Starting intervention learning data collection. Data directory: ${DATA_DIRECTORY}"

    LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
    CUDA_VISIBLE_DEVICES=$INTERVENTION_GPU \
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

    >&2 echo "Data collection stopped"
    ;;
esac
