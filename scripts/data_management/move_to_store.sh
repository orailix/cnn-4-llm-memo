#!/bin/bash

# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

# To remove for a job on `archive` partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread

# To keep
#SBATCH --job-name=tar_output
#SBATCH --partition=archive
#SBATCH --ntasks=1
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --account=yfw@cpu

# This job is intended to archive the output of the computation to a storage unit.
# Edit the paths accordingly.

SOURCE_DIR="/lustre/fswork/projects/rech/yfw/upp42qa/detect-regurgitation/output"
DEST_DIR="/lustre/fsstor/projects/rech/yfw/upp42qa"
ARCHIVE_NAME="output_regu_detect_$(date +'%Y_%m_%d__%H_%M_%S').tar"

# Create the tar archive
echo "Starting backup of $SOURCE_DIR to $DEST_DIR/$ARCHIVE_NAME"
tar -cf "$DEST_DIR/$ARCHIVE_NAME" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

# Check if the command was successful
if [ $? -eq 0 ]; then
  echo "Backup completed successfully."
else
  echo "Backup failed."
  exit 1
fi
