#!/bin/bash

# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

#SBATCH --array=0-80
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
    "merge_2_other_guess_recollect_recite"
)

# Possible ROUGE_THRESHOLD
export ROUGE_THRESHOLD_ALL=(
    "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9"
    "1.1" "1.2" "1.3" "1.4" "1.5" "1.6" "1.7" "1.8" "1.9"
    "2.1" "2.2" "2.3" "2.4" "2.5" "2.6" "2.7" "2.8" "2.9"
)

# Possible model_size
export SIZE_ALL=("12b" "6.9b" "2.8b")

# Config - adaptable
TASK_IDX=${SLURM_ARRAY_TASK_ID:-0}

TAX_NAME_IDX=$((TASK_IDX % ${#TAX_NAME_ALL[@]}))
export TAX_NAME="${TAX_NAME_ALL[$TAX_NAME_IDX]}"
TASK_IDX=$((TASK_IDX / ${#TAX_NAME_ALL[@]}))

ROUGE_THRESHOLD_IDX=$((TASK_IDX % ${#ROUGE_THRESHOLD_ALL[@]}))
export ROUGE_THRESHOLD="${ROUGE_THRESHOLD_ALL[$ROUGE_THRESHOLD_IDX]}"
TASK_IDX=$((TASK_IDX / ${#ROUGE_THRESHOLD_ALL[@]}))

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
export WANDB_GROUP="optimize_rho"

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
