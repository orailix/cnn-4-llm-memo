#!/bin/bash

# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

#SBATCH --array=0-179
#SBATCH --account=yfw@a100
#SBATCH --job-name=allsize
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=/lustre/fswork/projects/rech/yfw/upp42qa/detect-regurgitation/logs/%x.%A_%a.out
#SBATCH --error=/lustre/fswork/projects/rech/yfw/upp42qa/detect-regurgitation/logs/%x.%A_%a.out
#SBATCH --no-requeue

# =========================================================
# ======================= TAX NAME ========================
# =========================================================

# Possible taxonomies
export TAX_NAME_ALL=(

    ################### RECONSTRUCT ###################

    ### 4 CATEGORIES

    "other_recite_reconstruct_recollect" # -> OK
    # "other_recollect_recite_reconstruct" -> NOK, only 3 categories
    # "other_reconstruct_recollect_recite" -> NOK, same as the one below because order for "recite" and "recollect" does not matter
    "other_reconstruct_recite_recollect" # -> OK
    "other_recollect_reconstruct_recite" # -> OK
    # "other_recite_recollect_reconstruct" -> NOK, only 3 categories

    ### 3 CATEGORIES

    "merge_1_other_recite_reconstruct_recollect" # -> OK
    # "merge_1_other_recollect_recite_reconstruct" -> NOK, got everyone in first category
    "merge_1_other_reconstruct_recollect_recite" # -> OK
    # "merge_1_other_reconstruct_recite_recollect" -> NOK, same as first of this batch
    # "merge_1_other_recollect_reconstruct_recite" -> NOK, same as the 3rd of this batch
    # "merge_1_other_recite_recollect_reconstruct" -> NOK, got everyone in this category

    "merge_2_other_recite_reconstruct_recollect" # -> OK
    "merge_2_other_recollect_recite_reconstruct" # -> OK
    "merge_2_other_reconstruct_recollect_recite" # -> OK
    # "merge_2_other_reconstruct_recite_recollect" -> NOK, same as the 3rd of the batch
    # "merge_2_other_recollect_reconstruct_recite" -> NOK, same as the 2nd of the batch
    # "merge_2_other_recite_recollect_reconstruct" -> NOK? same as the 1st of the batch

    ################### GUESS ###################

    ### 4 CATEGORIES

    "other_recite_guess_recollect" # -> OK
    "other_guess_recite_recollect" # -> OK
    "other_recollect_guess_recite" # -> OK

    ### 3 CATEGORIES

    "merge_1_other_recite_guess_recollect" # -> OK
    "merge_1_other_guess_recollect_recite" # -> OK
    # "merge_2_other_recite_guess_recollect" # -> NOK, same as merge_2_other_recite_reconstruct_recollect
    # "merge_2_other_recollect_recite_guess" # -> NOK, same as merge_2_other_recollect_recite_reconstruct
    "merge_2_other_guess_recollect_recite" # -> OK

    ################### CODE ####################

    ### 4 CATEGORIES

    "other_recite_code_recollect" # -> OK
    "other_code_recite_recollect" # -> OK
    "other_recollect_code_recite" # -> OK

    ### 3 CATEGORIES

    "merge_1_other_recite_code_recollect" # -> OK
    "merge_1_other_code_recollect_recite" # -> OK
    "merge_2_other_code_recollect_recite" # -> OK
)

# Possible duplicates_threshold
export DUPLICATES_THRESHOLD_ALL=("5" "50" "1000")

# Possible model_size
export SIZE_ALL=("12b" "6.9b" "2.8b")

# Config - adaptable
TASK_IDX=${SLURM_ARRAY_TASK_ID:-0}

TAX_NAME_IDX=$((TASK_IDX % ${#TAX_NAME_ALL[@]}))
export TAX_NAME="${TAX_NAME_ALL[$TAX_NAME_IDX]}"
TASK_IDX=$((TASK_IDX / ${#TAX_NAME_ALL[@]}))

DUPLICATES_THRESHOLD_IDX=$((TASK_IDX % ${#DUPLICATES_THRESHOLD_ALL[@]}))
export DUPLICATES_THRESHOLD="${DUPLICATES_THRESHOLD_ALL[$DUPLICATES_THRESHOLD_IDX]}"
TASK_IDX=$((TASK_IDX / ${#DUPLICATES_THRESHOLD_ALL[@]}))

export SIZE="${SIZE_ALL[$TASK_IDX]}"

# =========================================================
# ======================== SETUP ==========================
# =========================================================

# Paths
ROOT=/lustre/fswork/projects/rech/yfw/upp42qa/detect-regurgitation
export _BASE_OUTPUT_DIR=/lustre/fsn1/projects/rech/yfw/upp42qa/output_regu_detect

# Setup env
source ~/.bashrc
conda activate regu
cd $ROOT

# =========================================================
# =================== COMPUTE PATTERNS ====================
# =========================================================

# Config
export TRAIN_BASE_SIZE=4000
export EVAL_BASE_SIZE=2000
export INFERENCE_BS=16

# COMPUTE PATTERNS
python -u -m src compute-patterns

# =========================================================
# ====================== TRAIN CNNs =======================
# =========================================================

# GET CONFIG HASH
export PATTERNS_CONFIG=$(python -u -m src get-current-config-hash)

# Config - fixed
export TRAIN_BS=16
export N_EPOCHS=3
export NUM_WORKERS=2
export PREFETCH_FACTOR=16

# Wandb group
export WANDB_PROJECT="regu_detect"
export WANDB_GROUP=SIZE

# Variable config
export KERNEL_SIZE_ALL=("6" "8")
export N_FEAT_CNN_ALL=("10" "16")
export N_FEAT_FC="64"
export HEAD_POOLING_ALL=("mean" "max")

# Code execution
for KERNEL_SIZE in "${KERNEL_SIZE_ALL[@]}"; do
    for N_FEAT_CNN in "${N_FEAT_CNN_ALL[@]}"; do
        for HEAD_POOLING in "${HEAD_POOLING_ALL[@]}"; do
            export KERNEL_SIZE N_FEAT_CNN HEAD_POOLING
            python -u -m src train
        done
    done
done
