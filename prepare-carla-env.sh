#!/usr/bin/env bash

. ./utils.sh

export XDG_RUNTIME_DIR=/tmp
export DISPLAY=
export INTERVENTION_CARLA_SOURCE_DIRECTORY=~/carla/CARLA_0.9.10.1_RSS
export INTERVENTION_CARLA_DIRECTORY=/tmp/CARLA_0.9.10.1_RSS-7a916435-0cf2-4e4a-a595-6bf6a6e0cb90

CARLA_HASH="9bcfd1021e04366fd3356588ee3b09213069cb888e7ef4c574e406e0cdd9c16e"

(
    flock -s 200
    echo >&2 "Checking Carla simulator checksum"
    hash=$(tree_sha256_hash "${INTERVENTION_CARLA_DIRECTORY}")

    if [ $? -eq 0 ] && [ "${hash}" == "${CARLA_HASH}" ]; then
        echo >&2 "Carla simulator checksum matches"
    else
        echo >&2 "Carla simulator checksum does not match."
        echo >&2 "    Expected: ${CARLA_HASH}"
        echo >&2 "    Got:      ${hash}"
        echo >&2 "Copying Carla simulator from ${INTERVENTION_CARLA_SOURCE_DIRECTORY} to cluster-local filesystem ${INTERVENTION_CARLA_DIRECTORY}"
        rsync -r "${INTERVENTION_CARLA_SOURCE_DIRECTORY}/" "${INTERVENTION_CARLA_DIRECTORY}"
    fi
) 200>"${INTERVENTION_CARLA_DIRECTORY}.lock"

export CARLA_WORLD_PORT=$(get_available_carla_port)
