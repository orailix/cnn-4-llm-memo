# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import functools
import typing as t
from time import time

import h5py
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from transformer_lens import HookedTransformer

from ..patterns import PatternsConfig
from ..patterns.models import get_default_tokenizer

COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
tokenizer = get_default_tokenizer()


@functools.lru_cache()
def get_newline_tokens() -> torch.Tensor:
    """Gets the indices of tokens that contain "\n" in their string form."""
    result = []
    for tok in range(len(tokenizer)):
        if "\n" in tokenizer.decode(tok):
            result.append(tok)

    return torch.Tensor(result).to(dtype=int)


def get_patterns_for_repr(
    df: pd.DataFrame,
    patterns_cfg: PatternsConfig,
    loc: int,
):
    # Getting sequence id
    sequence_id = df.loc[loc, "sequence_id"]

    # Getting pattertns
    with h5py.File(patterns_cfg.get_patterns_path(), "r") as h5f:
        attention_pattern = h5f[str(sequence_id)][()]  # Load as np array

    # Converting to torch
    attention_pattern = torch.Tensor(attention_pattern)

    # Setting value of newline to zero
    # newline_tokens = get_newline_tokens()
    # newline_token_mask = torch.isin(torch.Tensor(df.loc[loc, "tokens"].copy()).long(), newline_tokens)
    # newline_indices = newline_token_mask.nonzero(as_tuple=False)
    # tok_idx = newline_indices[:, 0]
    # attention_pattern[:, :, :, tok_idx] = 0
    # attention_pattern[..., torch.triu(torch.ones(64, 64, dtype=bool), diagonal=1)] = 0

    return attention_pattern


def plot_attention_pattern(
    attention_pattern: torch.Tensor,
    ax: plt.Axes,
    head_pooling: t.Optional[str] = None,
    color_idx: t.Optional[int] = None,
    alpha: t.Optional[float] = None,
    gamma: float = 1 / 5,
    add_diag: bool = True,
    show_axlabels: bool = False,
) -> None:
    """Plots the attention pattern on a given layer.

    Parameters
    ----------
        attention_pattern :
            A tensor of shape (num_head, num_token, num_token) containing the attention pattern.
        ax :
            The ax to plot the attention pattern on
        head_pooling :
            Either `None` (no pooling of the heads) or `mean` or `median` or `min` or `max` or `std`
        colors_idx :
            Should be non-None only if there is a single head.
        alpha :
            The alpha to be passed to plt.imshow.
        gamma :
            The gamme to be passed to LinearSegmentedColormap
        add_diag :
            Whether or not to add a diagonal to each plot to spot its position
        show_axlabels :
            Whether or not to show the labels "Key token idx" and "Query token idx" as ax labels
    """

    # Reshape
    attention_pattern = attention_pattern.cpu()
    if attention_pattern.dim() == 2:
        attention_pattern = attention_pattern.view(
            -1, attention_pattern.size(0), attention_pattern.size(1)
        )

    # Head pooling
    if head_pooling == "mean":
        attention_pattern = torch.mean(attention_pattern, dim=0, keepdim=True)
    elif head_pooling == "median":
        attention_pattern = torch.median(attention_pattern, dim=0, keepdim=True)
    elif head_pooling == "max":
        attention_pattern = torch.max(attention_pattern, dim=0, keepdim=True).values
    elif head_pooling == "min":
        attention_pattern = torch.min(attention_pattern, dim=0, keepdim=True).values
    elif head_pooling == "std":
        attention_pattern = torch.std(attention_pattern, dim=0, keepdim=True)

    # Num heads
    num_head, _, _ = attention_pattern.size()
    num_colors = num_head if color_idx is None else max(num_head, color_idx)
    colors = (num_colors // len(COLORS) + 1) * COLORS

    for head in range(num_head):
        if head == 0:
            color_idx = head if color_idx is None else color_idx
        else:
            color_idx = head
        cmap = LinearSegmentedColormap.from_list(
            f"custom_{color_idx}_{time()}",
            ["#FFFFFF00", colors[color_idx]],
            gamma=gamma,
        )
        ax.imshow(
            attention_pattern[head, ...],
            cmap=cmap,
            alpha=alpha,
            vmin=0 if head_pooling in [None, "max"] else None,
            vmax=1 if head_pooling in [None, "max"] else None,
        )

        if add_diag and head == num_head - 1:
            height, width = attention_pattern[head, ...].size()
            ax.plot(
                [0, width - 1], [0, height - 1], color="black", linewidth=0.5, alpha=0.5
            )

    # Ax labels
    if show_axlabels:
        ax.set_xlabel("Key token index")
        ax.set_ylabel("Query token index")


def plot_singlesentence_multihead(
    attention_pattern: torch.Tensor,
    layer: int,
    max_row: int = 10,
    gamma: float = 1 / 5,
    head_pooling: t.Optional[str] = None,
    add_diag: bool = True,
    show_axlabels: bool = False,
) -> plt.Figure:
    """Plots all attention heads at a given layer."""

    attention_pattern = attention_pattern.cpu()
    num_head, _, _ = attention_pattern[layer].size()

    n_row = min(max_row, num_head)
    n_col = (num_head + (n_row - 1)) // n_row + 1
    fig, axs = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))

    # Heads
    for head in range(num_head):
        ax = axs[head % 10, head // 10]
        plot_attention_pattern(
            attention_pattern[layer, head, ...],
            color_idx=head,
            ax=ax,
            alpha=None,
            gamma=gamma,
            head_pooling=head_pooling,
            add_diag=add_diag,
            show_axlabels=show_axlabels,
        )
        ax.set_title(f"Layer {layer} Head {head}")

    # Full
    ax = axs[0, -1]
    plot_attention_pattern(
        attention_pattern[layer, ...],
        ax=ax,
        alpha=None,
        gamma=gamma,
        head_pooling=head_pooling,
        add_diag=add_diag,
        show_axlabels=show_axlabels,
    )
    ax.set_title(f"Layer {layer} FULL")

    for idx in range(1, n_row):
        axs[idx, 4].axis("off")

    fig.tight_layout()

    return fig


def plot_multisentence_multilayer(
    attention_pattern: torch.Tensor,
    layers: t.Optional[t.List[int]] = None,
    head_pooling: t.Optional[str] = None,
    gamma: float = 1 / 5,
    add_diag: bool = True,
    titles: t.Optional[t.List[str]] = None,
    show_axlabels: bool = False,
    layers_offset_in_title: int = 0,
) -> plt.Figure:
    """Plots multiple sentences at multiple layers."""

    # Need batching?
    if attention_pattern.dim() == 4:
        attention_pattern = attention_pattern[None, ...]

    bs, n_layer, _, _, _ = attention_pattern.size()

    if layers == None:
        layers = list(range(n_layer))[::-1]

    n_row, n_col = len(layers), bs
    fig, axs = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))

    for idx_col in range(n_col):
        for idx_row in range(n_row):

            if n_col == 1 and n_row == 1:
                ax = axs
            elif n_col == 1:
                ax = axs[idx_row]
            elif n_row == 1:
                ax = axs[idx_col]
            else:
                ax = axs[idx_row, idx_col]

            l = layers[idx_row]

            plot_attention_pattern(
                attention_pattern[idx_col, l, ...],
                ax,
                head_pooling=head_pooling,
                gamma=gamma,
                add_diag=add_diag,
                show_axlabels=show_axlabels,
            )

            if titles is None:
                ax.set_title(f"Sentence {idx_col} Layer {l + layers_offset_in_title}")
            else:
                ax.set_title(f"{titles[idx_col]} Layer {l + layers_offset_in_title}")

    # Output
    fig.tight_layout()
    return fig
