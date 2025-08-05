# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import functools
import typing as t

import torch


@functools.lru_cache()
def get_source_target_up(num_token: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    source = torch.tensor(
        [[row, col] for col in range(num_token) for row in range(col, num_token)]
    )
    target = torch.tensor(
        [[row, col] for col in range(num_token) for row in range(0, num_token - col)]
    )

    return source, target


@functools.lru_cache()
def get_source_target_right(num_token: int) -> t.Tuple[torch.Tensor, torch.Tensor]:
    source = torch.tensor(
        [[row, col] for row in range(num_token) for col in range(0, row + 1)]
    )
    target = torch.tensor(
        [
            [row, col]
            for row in range(num_token)
            for col in range(num_token - row - 1, num_token)
        ]
    )

    return source, target


def stack_to_up(attention_pattern: torch.Tensor) -> torch.Tensor:
    """TO BE COMPLETED"""

    _, _, _, _, num_token = attention_pattern.size()
    source, target = get_source_target_up(num_token)
    result = torch.zeros_like(attention_pattern, device="cpu")
    result[:, :, :, target[:, 0], target[:, 1]] = attention_pattern[
        :, :, :, source[:, 0], source[:, 1]
    ]

    return result


def stack_to_right(attention_pattern: torch.Tensor) -> torch.Tensor:
    """TO BE COMPLETED"""

    _, _, _, _, num_token = attention_pattern.size()
    source, target = get_source_target_right(num_token)
    result = torch.zeros_like(attention_pattern, device="cpu")
    result[:, :, :, target[:, 0], target[:, 1]] = attention_pattern[
        :, :, :, source[:, 0], source[:, 1]
    ]

    return result
