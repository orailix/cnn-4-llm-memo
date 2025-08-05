# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t
from dataclasses import dataclass
from pathlib import Path

from ..utils import BaseConfig


@dataclass
class PatternsConfig(BaseConfig):
    @classmethod
    def key_ignored_in_hash(cls) -> t.List[str]:
        return [
            "inference_bs",
            "_base_output_dir",
        ]

    @classmethod
    def class_output_name(cls) -> str:
        return "patterns"

    def __repr__(self):
        return super().__repr__()

    def get_patterns_path(self) -> Path:
        return self.get_output_dir() / "attention_patterns.h5py"

    # Model choice
    size: str = "12b"
    deduped: bool = True

    # Dataset
    seed: int = 42
    tax_name: str = ""
    rouge_threshold: float = 0.5
    duplicates_threshold: int = 5
    train_base_size: int = 10_000
    eval_base_size: int = 4_000
    inference_bs: int = 96
