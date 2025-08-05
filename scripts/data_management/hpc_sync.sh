#!/bin/bash

# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

# This is an example to sync the output of the computations from an HPC cluster to your local computer.
# You should edit the HOST, REMOTE_PATHS and LOCAL_PATH according to your own configuration
# Moving to local computer enables you to run most of the scripts in `figures` locally instead of HPC cluster

# Vars
HOST=jeanzay
REMOTE_PATHS=(
    "/lustre/fsn1/projects/rech/yfw/upp42qa/output_regu_detect/"
    "/lustre/fswork/projects/rech/yfw/upp42qa/detect-regurgitation/output/"
)
LOCAL_PATH=/Users/jeremie/Documents/01-Travail/01-Doctorat/regu-detect/output/

# Logging
echo "Syncing from multiple remote sources to $LOCAL_PATH ..."

for REMOTE_PATH in "${REMOTE_PATHS[@]}"; do
    echo "rsync ${REMOTE_PATH} >> ${LOCAL_PATH} ..."
    rsync \
        -zvar \
        --include='/*' \
        --exclude='*.h5py' \
        $HOST:$REMOTE_PATH $LOCAL_PATH
done

# Logging
REMOTE_PATHS=(
    "/lustre/fswork/projects/rech/yfw/upp42qa/detect-regurgitation/output/wandb/"
)
LOCAL_PATH=/Users/jeremie/Documents/01-Travail/01-Doctorat/regu-detect/output/wandb/
echo "Syncing wandb..."

for REMOTE_PATH in "${REMOTE_PATHS[@]}"; do
    echo "rsync ${REMOTE_PATH} >> ${LOCAL_PATH} ..."
    rsync \
        -zvar \
        --include='/*' \
        --exclude='*.h5py' \
        $HOST:$REMOTE_PATH $LOCAL_PATH
done
