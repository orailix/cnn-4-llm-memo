# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import functools
import re
import typing as t
from typing import List

import fasttext
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger
from rouge_score import rouge_scorer
from tqdm import tqdm

from ..utils import paths
from .config import PatternsConfig
from .generate import generate_base
from .models import get_model


@functools.lru_cache()
def get_full_dataset() -> DatasetDict:
    return load_dataset("usvsnsp/semantic-filters")


def get_memories(
    cfg: PatternsConfig,
    add_rouge_cols: bool = True,
) -> pd.DataFrame:
    """Get the dataset/dataframe of memorized samples, as well as its identifier."""

    # Setup
    full_ds = get_full_dataset()
    name = f"memories_{'deduped' if cfg.deduped else 'duped'}_{cfg.size}"

    # Rouge cols?
    if add_rouge_cols:
        df_without_rouge = get_memories(cfg, add_rouge_cols=False)
        df_with_rouge = add_rouge_cols_to_df(df_without_rouge, name)

        return df_with_rouge

    # Without rouge
    output = full_ds[name]
    output = output.to_pandas(batched=False)

    return output


def get_representatives(
    cfg: PatternsConfig,
    add_rouge_cols: bool = True,
) -> t.Tuple[t.Union[Dataset, pd.DataFrame], str]:
    """Get the dataset/dataframe of random samples, as well as its identifier."""

    # Get output
    full_ds = get_full_dataset()
    name = f"pile_{'deduped' if cfg.deduped else 'duped'}_{cfg.size}"

    # Rouge cols?
    if add_rouge_cols:
        df_without_rouge = get_representatives(cfg, add_rouge_cols=False)
        df_with_rouge = add_rouge_cols_to_df(df_without_rouge, name)

        return df_with_rouge

    # Without rouge
    output = full_ds[name]
    output = output.to_pandas(batched=False)

    return output


def get_rouge_df(
    df: pd.DataFrame,
    name: t.Optional[str] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """If the rouge score has already been computed, fetches it in the cache.

    Returns a dataset containing only two columns:
    - rouge_3_prefix_to_suffix
    - rouge_L_prefix_to_suffix
    """

    # Cache path
    rouge_cache_path = paths.cache_path / "rouge"
    rouge_cache_path.mkdir(exist_ok=True, parents=False)
    file_path = rouge_cache_path / f"{name}.parquet"

    # Looking for cache
    if name is not None and not force_recompute:
        logger.info(f"Looking in cache: {rouge_cache_path}")

        if not file_path.is_file():
            logger.info(f"Did not find {file_path}, computing it")
        else:
            logger.info(f"Reading ROUGE values from {file_path}")
            return pd.read_parquet(file_path)
    else:
        logger.info(f"Ignoring cache")

    # If not found, computing the result
    scorer_3 = rouge_scorer.RougeScorer(["rouge3"], use_stemmer=False)
    scorer_L = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    def process_column_rouge_3(val):
        reference = " ".join(map(str, val[:32]))
        candidate = " ".join(map(str, val[32:]))
        score = scorer_3.score(reference, candidate)

        return score["rouge3"].fmeasure

    def process_column_rouge_L(val):
        reference = " ".join(map(str, val[:32]))
        candidate = " ".join(map(str, val[32:]))
        score = scorer_L.score(reference, candidate)

        return score["rougeL"].fmeasure

    # Computation core
    logger.debug(f"Computing ROUGE-3...")
    result = pd.DataFrame(
        {"rouge_3_prefix_to_suffix": df["tokens"].map(process_column_rouge_3)}
    )
    logger.debug(f"Computing ROUGE-L...")
    result["rouge_L_prefix_to_suffix"] = df["tokens"].map(process_column_rouge_L)
    logger.debug(f"ROUGE computation finished!")

    # Saving to cache
    logger.debug(f"Caching at {file_path}")
    result.to_parquet(file_path)

    # Output
    return result


def get_is_code_df(
    df: pd.DataFrame,
    name: t.Optional[str] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """If the is_code score has already been computed, fetches it in the cache.

    Returns a dataset containing only one columns:
    - is_code: a bool representing if a sample is code or not
    """

    # Cache path
    is_code_cache_path = paths.cache_path / "is_code"
    is_code_cache_path.mkdir(exist_ok=True, parents=False)
    file_path = is_code_cache_path / f"{name}.parquet"

    # Looking for cache
    if name is not None and not force_recompute:
        logger.info(f"Looking in cache: {is_code_cache_path}")

        if not file_path.is_file():
            logger.info(f"Did not find {file_path}, computing it")
        else:
            logger.info(f"Reading IS_CODE values from {file_path}")
            return pd.read_parquet(file_path)
    else:
        logger.info(f"Ignoring cache")

    # If not found, computing the result
    model_path = hf_hub_download(
        repo_id="kenhktsui/code-natural-language-fasttext-classifier",
        filename="model.bin",
    )

    # Load FastText model
    model = fasttext.load_model(model_path)

    def replace_newlines(text: str) -> str:
        return re.sub("\n+", " ", text)

    def predict(text_list: List[str]) -> List[float]:
        text_list = [replace_newlines(text) for text in text_list]
        pred = model.predict(text_list)
        model_detect = [(l[0].lstrip("__label__") == "Code") for l, _ in zip(*pred)]
        latex_detect = ["\\usepackage{" in item for item in text_list]
        return [(mod_p or lat_p) for mod_p, lat_p in zip(model_detect, latex_detect)]

    # Computation core
    logger.debug(f"Computing IS_CODE...")
    result = pd.DataFrame({"is_code": predict(df["text"].tolist())})
    logger.debug(f"IS_CODE computation finished!")

    # Saving to cache
    logger.debug(f"Caching at {file_path}")
    result.to_parquet(file_path)

    # Output
    return result


def add_rouge_cols_to_df(
    df: pd.DataFrame,
    name: t.Optional[str] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Calls `get_rouge_df` and adds the resulting columns as
    "rouge_3_prefix_to_suffix" and "rouge_L_prefix_to_suffix"."""

    rouge_cols = get_rouge_df(df, name, force_recompute)
    is_code_col = get_is_code_df(df, name, force_recompute)
    df["rouge_3_prefix_to_suffix"] = rouge_cols["rouge_3_prefix_to_suffix"]
    df["rouge_L_prefix_to_suffix"] = rouge_cols["rouge_L_prefix_to_suffix"]
    df["is_code"] = is_code_col["is_code"]

    return df


def merge_1(where_other, where_cat1, where_cat2, where_cat3):
    return (where_other, np.concatenate([where_cat1, where_cat2]), where_cat3)


def merge_2(where_other, where_cat1, where_cat2, where_cat3):
    return (where_other, where_cat1, np.concatenate([where_cat2, where_cat3]))


def get_primary_category(
    df: pd.DataFrame,
    category: str,
    rouge_threshold: float,
    duplicates_threshold: int,
) -> pd.Series:

    if category == "other":
        return df["memorization_score"] < 1

    if category == "recite":
        return (df["memorization_score"] == 1) & (
            df["sequence_duplicates"] > duplicates_threshold
        )

    if category == "reconstruct":
        return (df["memorization_score"] == 1) & (
            df["is_repeating"] | df["is_incrementing"]
        )

    if category == "guess":
        if 0.0 < rouge_threshold and rouge_threshold <= 1.0:
            return (df["memorization_score"] == 1) & (
                df["is_repeating"]
                | df["is_incrementing"]
                | (df["rouge_L_prefix_to_suffix"] >= rouge_threshold)
                | (df["rouge_3_prefix_to_suffix"] >= rouge_threshold)
            )

        elif 1.0 < rouge_threshold and rouge_threshold <= 2.0:
            return (df["memorization_score"] == 1) & (
                df["is_repeating"]
                | df["is_incrementing"]
                | (df["rouge_L_prefix_to_suffix"] >= (rouge_threshold - 1.0))
            )

        elif 2.0 < rouge_threshold and rouge_threshold <= 3.0:
            return (df["memorization_score"] == 1) & (
                df["is_repeating"]
                | df["is_incrementing"]
                | (df["rouge_3_prefix_to_suffix"] >= (rouge_threshold - 2.0))
            )

        else:
            raise ValueError(f"Unrocognized rouge_threshold: {rouge_threshold}")

    if category == "recollect":
        return (df["memorization_score"] == 1) & (
            df["sequence_duplicates"] <= duplicates_threshold
        )

    if category == "memo":
        return df["memorization_score"] == 1

    if category == "code":
        return (df["memorization_score"] == 1) & (df["is_code"])

    raise ValueError(
        f"category={category} should be in ['other', 'recite', 'reconstruct', 'guess', 'recollect', 'code', 'memo']"
    )


def get_taxonomy_by_name(
    df: pd.DataFrame,
    tax_name: str,
    rouge_threshold: float,
    duplicates_threshold: int,
):

    # Merge 1
    if tax_name[: len("merge_1_")] == "merge_1_":
        where_other, where_cat1, where_cat2, where_cat3 = get_taxonomy_by_name(
            df,
            tax_name[len("merge_1_") :],
            rouge_threshold,
            duplicates_threshold,
        )
        return merge_1(where_other, where_cat1, where_cat2, where_cat3)

    # Merge 2
    if tax_name[: len("merge_2_")] == "merge_2_":
        where_other, where_cat1, where_cat2, where_cat3 = get_taxonomy_by_name(
            df,
            tax_name[len("merge_2_") :],
            rouge_threshold,
            duplicates_threshold,
        )
        return merge_2(where_other, where_cat1, where_cat2, where_cat3)

    # Categories
    primary_categories = tax_name.split("_")
    if len(primary_categories) != 4:
        raise ValueError(f"Taxonomy name not recognized: {tax_name}")

    is_other, is_cat_0, is_cat_1, is_cat_2 = (
        get_primary_category(df, cat, rouge_threshold, duplicates_threshold)
        for cat in primary_categories
    )

    is_cat_1 = is_cat_1 & (~is_cat_0)
    is_cat_2 = is_cat_2 & (~is_cat_0) & (~is_cat_1)

    return (
        np.where(is_other)[0],
        np.where(is_cat_0)[0],
        np.where(is_cat_1)[0],
        np.where(is_cat_2)[0],
    )


def get_train_eval_test(
    cfg: PatternsConfig,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """Returns a pandas DataFrame containing a train, eval and test set.

    Two columns for one-hot encoding of the partition:
    - "part_train"
    - "part_eval"

    Some columns for one-hot encoding of the category:
    - "cat_0" for "other" (non-memorized with free continuation
    - "cat_1", depending on the taxonomy
    - "cat_2", depending on the taxonomy
    - etc.

    For each of the four categories, we sample `train_base_size` train elements,
    `eval_base_size` eval elements.

    The seed ensures reproducibility.
    """

    # Caching?
    file_path = cfg.get_output_dir() / "train_eval_test.parquet"
    if not force_recompute:
        logger.info(f"Looking in cache: {file_path}")

        if not file_path.is_file():
            logger.info(f"Did not find {file_path}, computing it")
        else:
            logger.info(f"Reading dataset from {file_path}")
            return pd.read_parquet(file_path)
    else:
        logger.info(f"Ignoring cache")

    # Getting repz and memo df
    repz_df = get_representatives(cfg, add_rouge_cols=True)
    memo_df = get_memories(cfg, add_rouge_cols=True)
    combo_df = pd.concat([repz_df, memo_df])
    combo_df = combo_df.drop_duplicates(subset="sequence_id")

    # Getting indices of all elements
    categories = get_taxonomy_by_name(
        combo_df,
        tax_name=cfg.tax_name,
        rouge_threshold=cfg.rouge_threshold,
        duplicates_threshold=cfg.duplicates_threshold,
    )

    # Sampling
    num_caterogies = len(categories)
    categories = [sorted(cat) for cat in categories]

    rg = np.random.RandomState(cfg.seed)
    to_be_concatenated = []
    for idx_cat, cat_container in enumerate(categories):
        # Full selection
        full_size = cfg.train_base_size + cfg.eval_base_size
        full_selected = rg.choice(cat_container, size=full_size, replace=False)
        full_selected_df_view = combo_df.iloc[full_selected].copy()

        # Adding one-hot encoding
        for idx_onehot in range(num_caterogies):
            full_selected_df_view[f"cat_{idx_onehot}"] = (
                True if idx_cat == idx_onehot else False
            )

        # Train and eval one-hot
        full_selected_df_view["part_train"] = False
        full_selected_df_view["part_eval"] = False

        # Train
        train_selection = full_selected_df_view.iloc[: cfg.train_base_size].copy()
        train_selection["part_train"] = True
        to_be_concatenated.append(train_selection)

        # Eval
        eval_selection = full_selected_df_view.iloc[
            cfg.train_base_size : cfg.train_base_size + cfg.eval_base_size
        ].copy()
        eval_selection["part_eval"] = True
        to_be_concatenated.append(eval_selection)

    # The first two elements must have free generation, because they are "other"
    logger.info(f"Computing the free generation for 'other' category")
    model = get_model(cfg, hooked=False, eager_attention=False)

    for other_container in to_be_concatenated[:2]:

        # Iterating over samples
        np_tokens = other_container["tokens"].to_numpy()
        np_tokens = np.stack(np_tokens)
        ds_len = len(other_container)
        bs = cfg.inference_bs
        num_batch = int(np.ceil(ds_len / bs))
        for b in tqdm(range(num_batch)):
            prompt = (
                torch.Tensor(np_tokens[bs * b : bs * (b + 1), :32])
                .int()
                .to(device=model.device)
                .view(-1, 32)
            )
            output = generate_base(model, prompt)
            np_tokens[bs * b : bs * (b + 1), 32:] = (
                output["sequences"][:, 32:].cpu().detach().numpy()
            )

        other_container["tokens"] = list(np_tokens)  # List of np.ndarray

    # Concatenating
    result = pd.concat(to_be_concatenated, ignore_index=True)

    # Cleaning
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Caching
    logger.debug(f"Caching at {file_path}")
    result.to_parquet(file_path)

    # Output
    return result
