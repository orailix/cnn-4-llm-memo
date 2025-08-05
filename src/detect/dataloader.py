# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

from .config import DetectConfig


class H5Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_file_path: Path,
        keys: t.List[int],
        labels: t.List[int],
    ):
        self.h5_file_path = h5_file_path
        self.keys = keys
        self.labels = labels

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        label = self.labels[idx]
        with h5py.File(self.h5_file_path, "r") as h5f:
            patterns = h5f[str(key)][()]  # Load the tensor as numpy array

        return key, patterns, label


def collate_fn(batch):
    batched_keys = torch.stack([torch.tensor(item[0]) for item in batch])
    batched_patterns = torch.stack([torch.tensor(item[1]) for item in batch])
    batched_labels = torch.stack([torch.tensor(item[2]) for item in batch])

    return batched_keys, batched_patterns, batched_labels


def get_dataloader(
    dataset: H5Dataset,
    detect_cfg: DetectConfig,
    persistent_workers: bool = True,
    eval: bool = False,
):
    return DataLoader(
        dataset,
        shuffle=True if not eval else False,
        batch_size=detect_cfg.train_bs if not eval else detect_cfg.eval_bs,
        collate_fn=collate_fn,
        num_workers=detect_cfg.num_workers,
        prefetch_factor=detect_cfg.prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=True,
    )
