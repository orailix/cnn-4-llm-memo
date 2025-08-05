# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import functools
import typing as t

import torch
from loguru import logger
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast

from ..utils.constants import DEVICE, DTYPE
from .config import PatternsConfig

PYTHIA_SIZES = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
PYTHIA_SIZES_FLOAT = [
    float(val[:-1]) * 1e6 if val[-1] == "m" else float(val[:-1]) * 1e9
    for val in PYTHIA_SIZES
]


def get_model(
    cfg: PatternsConfig,
    hooked: bool = False,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    eager_attention: bool = False,
) -> t.Union[GPTNeoXForCausalLM, HookedTransformer]:
    """Gets Pythia model for evaluation."""
    if cfg.size not in PYTHIA_SIZES:
        raise ValueError(f"Got size={cfg.size}, should be in {PYTHIA_SIZES}")

    attn_kwarg = dict(attn_implementation="eager") if eager_attention else dict()
    if not eager_attention and hooked:
        logger.warning(
            f"Using `HookedTransformer` eager attention will be used anyways. Ignoring `eager_attention=False`."
        )

    if not hooked:
        model = GPTNeoXForCausalLM.from_pretrained(
            f"EleutherAI/pythia-{cfg.size}{'-deduped' if cfg.deduped else ''}-v0",
            torch_dtype=dtype,
            device_map=device,
            **attn_kwarg,
        )

    else:
        model = HookedTransformer.from_pretrained_no_processing(
            f"EleutherAI/pythia-{cfg.size}{'-deduped' if cfg.deduped else ''}-v0",
            device=device,
            dtype=dtype,
        )

    # Output
    model.eval()
    return model


def get_tokenizer(
    cfg: PatternsConfig,
) -> GPTNeoXTokenizerFast:
    """Gets Pythia Tokenizer for evaluation."""
    return AutoTokenizer.from_pretrained(
        f"EleutherAI/pythia-{cfg.size}{'-deduped' if cfg.deduped else ''}-v0",
        revision="step143000",
    )


@functools.lru_cache()
def get_default_tokenizer():
    return get_tokenizer(cfg=PatternsConfig())
