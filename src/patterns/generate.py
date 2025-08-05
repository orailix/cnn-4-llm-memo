# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers.generation import GenerateDecoderOnlyOutput

from .models import get_default_tokenizer


def generate_base(
    model: GPTNeoXForCausalLM,
    prompt: torch.Tensor,
    eos_token_id: int = 0,
) -> GenerateDecoderOnlyOutput:
    """Generates the 32 following tokens from a Pythia model.

    The attention is only outputed if the model has "eager" attention implementation.

    Parameters
    ----------
        model :
            The Pythia model to generate from
        prompt :
            A torch tensor of shape (bs, 32) containing the prompt
        eos_token_id :
            The token id of <eos>, used to auto-detect attention mask
    """
    # Sanity check
    if isinstance(model, HookedTransformer):
        raise TypeError(
            f"`model` should be instance of `GPTNeoXForCausalLM`. For `HookedTransformer`, please use `generate_hooked` function."
        )

    # Model eval
    model.eval()

    # Prompt pre-processing
    prompt = prompt.view(-1, 32).to(dtype=int, device=model.device)

    # Need output_attentions
    output_attention = "eager" in model.gpt_neox._attn_implementation

    with torch.no_grad():
        return model.generate(
            input_ids=prompt,
            do_sample=False,
            max_length=64,
            min_length=64,
            pad_token_id=eos_token_id,
            attention_mask=(prompt != eos_token_id).long(),
            output_hidden_states=True,
            output_scores=True,
            output_attentions=output_attention,
            return_dict_in_generate=True,
            return_legacy_cache=False,
        )


def get_repr(idx: int, sequence: pd.Series) -> str:
    elt = sequence.iloc[idx]
    quantile = np.sum(sequence >= elt) / len(sequence)
    if int(elt) == elt:
        return f"{elt:6} [{quantile:2.2%}]"
    else:
        return f"{elt:2.5} [{quantile:2.2%}]"


def repr_generation_diff(
    df: pd.DataFrame,
    idx: t.Union[int, t.List[int]],
) -> str:
    """Pretty representation of the representation difference.

    The DataFrame should be the output of get_train_eval_test(cfg)"""

    # Tokenizer
    tokenizer = get_default_tokenizer()

    # Batching if necessary
    if isinstance(idx, (int, np.integer)):
        idx = [idx]

    # key_to_loc
    key_to_loc = {}
    for loc in range(len(df)):
        key = df.loc[loc, "sequence_id"]
        key_to_loc[key] = loc

    # Generation
    generation_tokens = []
    for rank, loc in enumerate(idx):
        sequence_id = df.loc[loc, "sequence_id"]
        if sequence_id < 0:  # We have a "free generation"
            dataset_version_loc = key_to_loc[-1 * sequence_id]
            idx[rank] = dataset_version_loc
            df.loc[dataset_version_loc, "predicted_duplicates"] = df.loc[
                loc, "predicted_duplicates"
            ]
            df.loc[dataset_version_loc, "predicted_probabilities"] = df.loc[
                loc, "predicted_probabilities"
            ]
            generation_tokens.append(df.loc[loc, "tokens"][32:])
        elif -1 * sequence_id in key_to_loc:
            generation_loc = key_to_loc[-1 * sequence_id]
            generation_tokens.append(df.loc[generation_loc, "tokens"][32:])
        else:
            generation_tokens.append(None)

    # Tokens
    tokens = (
        torch.Tensor(np.stack(df.iloc[idx]["tokens"].to_numpy()))
        .view(-1, 64)
        .cpu()
        .int()
    )
    prefix = tokens[:, :32]
    suffix = tokens[:, 32:]

    # Getting facts about the element
    sequence_id = df["sequence_id"]
    sequence_duplicates = df["sequence_duplicates"]
    predicted_duplicates = df["predicted_duplicates"]
    memorization_score = df["memorization_score"]
    is_incrementing = df["is_incrementing"]
    is_repeating = df["is_repeating"]
    huffman_coding_length = df["huffman_coding_length"]
    prompt_perplexity = df["prompt_perplexity"]
    generation_perplexity = df["generation_perplexity"]
    sequence_perplexity = df["sequence_perplexity"]
    rouge_3_prefix_to_suffix = df["rouge_3_prefix_to_suffix"]
    rouge_L_prefix_to_suffix = df["rouge_L_prefix_to_suffix"]

    label = np.argmax(
        df[["cat_0", "cat_1", "cat_4"]].to_numpy() + df[["cat_0", "cat_1", "cat_3"]],
        axis=1,
    )
    predicted_probabilities = df["predicted_probabilities"]

    result = ""
    for rank in range(prefix.size(0)):
        idx_of_element = idx[rank]
        prefix_str = tokenizer.decode(prefix[rank])
        suffix_str = tokenizer.decode(suffix[rank])

        result += f"""
################################################################################

>>>  [ ID ] {sequence_id.iloc[idx_of_element]}
>>>  [MEMO] {get_repr(idx_of_element, memorization_score)}
>>>  [INCR] {get_repr(idx_of_element, is_incrementing)}
>>>  [REPT] {get_repr(idx_of_element, is_repeating)}
>>>  [HUFF] {get_repr(idx_of_element, huffman_coding_length)}
>>>  [P-PL] {get_repr(idx_of_element, prompt_perplexity)}
>>>  [G-PL] {get_repr(idx_of_element, generation_perplexity)}
>>>  [S-PL] {get_repr(idx_of_element, sequence_perplexity)}
>>>  [RG-3] {get_repr(idx_of_element, rouge_3_prefix_to_suffix)}
>>>  [RG-L] {get_repr(idx_of_element, rouge_L_prefix_to_suffix)}

>>>  [DUPL] {get_repr(idx_of_element, sequence_duplicates)}
>>>  [PR-D] {predicted_duplicates[idx_of_element]:2.5}

>>>  [CLAS] {label[idx_of_element]}
>>>  [PR-C] {predicted_probabilities[idx_of_element]}


>>>  [PREFIX]
{prefix_str}

>>>  [SUFFIX]
{suffix_str}
"""
        if generation_tokens[rank] is not None:
            generation_str = tokenizer.decode(generation_tokens[rank])
            result += f"""
>>>  [GENERATED]
{generation_str}
"""

    return result
