# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import base64
import hashlib
import typing as t
from dataclasses import dataclass, fields

from ..utils import BaseConfig


@dataclass
class DetectConfig(BaseConfig):
    @classmethod
    def key_ignored_in_hash(cls) -> t.List[str]:
        return [
            "num_workers",
            "prefetch_factor",
            "n_epochs",
            "train_bs",
            "eval_bs",
            "_base_output_dir",
        ]

    @classmethod
    def class_output_name(cls) -> str:
        return "detect"

    def __repr__(self):
        return super().__repr__()

    def detection_hash(self) -> str:
        to_hash = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if value != field.default:
                to_hash[field.name] = value

        # Hashing
        hashable = ""
        for key in sorted(to_hash):
            if key in self.__class__.key_ignored_in_hash():
                continue
            if key == "patterns_config":
                continue
            hashable += f"{key}={to_hash[key]}\n"

        # Persistent, replicable and URL-free hash
        return base64.urlsafe_b64encode(
            hashlib.md5(hashable.encode("utf-8")).digest()
        ).decode()[:22]

    # Patterns
    patterns_config: str = ""

    # Dataloaders
    num_workers: int = 3
    prefetch_factor: int = 2
    train_bs: int = 16
    eval_bs: int = 16

    # Model
    head_pooling: str = "mean"
    kernel_size: int = 8
    pooling_kernel: int = 2
    n_feat_cnn: int = 24
    n_feat_fc: int = 128

    # Training
    n_epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 0.1
