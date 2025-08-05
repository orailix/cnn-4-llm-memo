# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os
import typing as t
from datetime import datetime

import torch
import wandb
from loguru import logger
from torch import nn

from ..patterns import PatternsConfig, get_train_eval_test
from ..utils import paths
from ..utils.constants import CNN_DTYPE, DEVICE
from .cnn_2d import Cnn2D, train_cnn
from .config import DetectConfig
from .dataloader import H5Dataset, get_dataloader
from .utils import check_model_parameter_sizes


def train_cnn_2d(config: t.Optional[str]):
    # DetectConfig
    if config is None:
        logger.debug("Option --config is None, loading config from env variables")
        detect_cfg = DetectConfig.from_env()
    else:
        logger.debug(f"Got --config={config}, trying to load existing config")
        detect_cfg = DetectConfig.autoconfig(config)

    # Corresponding PatternsConfig
    patterns_cfg = PatternsConfig.autoconfig(detect_cfg.patterns_config)

    # Mkdir
    (detect_cfg.get_output_dir() / "cnn_2d").mkdir(exist_ok=True, parents=True)

    # Wandb
    additional_wandb_config = dict(
        detect_id=detect_cfg.get_id(),
        patterns_tax_name=patterns_cfg.tax_name,
        patterns_size=patterns_cfg.size,
        patterns_duplicates_threshold=patterns_cfg.duplicates_threshold,
        patterns_rouge_threshold=patterns_cfg.rouge_threshold,
    )
    wandb.init(
        dir=paths.output,
        project=os.getenv("WANDB_PROJECT", "cnn_2d"),
        group=os.getenv("WANDB_GROUP", None),
        config={
            "detection_hash": detect_cfg.detection_hash(),
            **detect_cfg.get_as_dict(),
            **additional_wandb_config,
        },
        mode="offline",
        name=f"{detect_cfg.get_id()[:4]}_on_{patterns_cfg.get_id()[:4]}__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}",
    )

    # Logger
    logger.info(f"Starting training with detect config: {detect_cfg}")
    logger.info(f"Underlying patterns config: {patterns_cfg}")

    # Getting train set
    logger.info("Loading train set")
    df = get_train_eval_test(patterns_cfg)

    # Auto number of classes
    num_classes = len([col for col in df.columns if "cat_" in col])
    logger.debug(f"Automatic dection of num classes: {num_classes}")

    # Dataloaders
    logger.debug("Initializing train dataloader")
    train_filter = df["part_train"]
    train_dataset = H5Dataset(
        h5_file_path=patterns_cfg.get_patterns_path(),
        keys=df[train_filter]["sequence_id"].tolist(),
        labels=df[train_filter][[f"cat_{idx}" for idx in range(num_classes)]]
        .to_numpy()
        .argmax(axis=1),
    )
    train_dataloader = get_dataloader(
        train_dataset, detect_cfg, persistent_workers=True
    )

    logger.debug("Initializing eval dataloader")
    eval_filter = df["part_eval"]
    eval_dataset = H5Dataset(
        h5_file_path=patterns_cfg.get_patterns_path(),
        keys=df[eval_filter]["sequence_id"].tolist(),
        labels=df[eval_filter][[f"cat_{idx}" for idx in range(num_classes)]]
        .to_numpy()
        .argmax(axis=1),
    )
    eval_dataloader = get_dataloader(eval_dataset, detect_cfg, persistent_workers=True)

    # Model
    logger.info("Loading model")
    model = Cnn2D(
        model_size=patterns_cfg.size,
        n_class=num_classes,
        kernel_size=detect_cfg.kernel_size,
        pooling_kernel=detect_cfg.pooling_kernel,
        n_feat_cnn=detect_cfg.n_feat_cnn,
        n_feat_fc=detect_cfg.n_feat_fc,
    ).to(dtype=CNN_DTYPE, device=DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=detect_cfg.lr, weight_decay=detect_cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    check_model_parameter_sizes(model)

    # Train
    logger.info("Starting training")
    train_cnn(
        model, train_dataloader, eval_dataloader, optimizer, criterion, detect_cfg
    )
