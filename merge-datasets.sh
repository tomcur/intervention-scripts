#!/usr/bin/env bash

# (Non-destructively) merges the dataset in argument 1 into the dataset in
# argument 2.  Cooperatively locks the target `episodes.csv`.

SOURCE=$1
TARGET=$2

[ ! -d "${TARGET}" ] && mkdir -p "${TARGET}"

(
    flock -s 201
    if [ -f "${TARGET}/episodes.csv" ]; then
        tail --lines=+2 "${SOURCE}/episodes.csv" >>"${TARGET}/episodes.csv"
    else
        cp "${SOURCE}/episodes.csv" "${TARGET}/episodes.csv"
    fi
) 201>"${TARGET}/episodes.csv.lock"

cp -r "${SOURCE}"/*/ "${TARGET}"
