# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

from .config import PatternsConfig
from .datasets import (
    add_rouge_cols_to_df,
    get_full_dataset,
    get_memories,
    get_representatives,
    get_rouge_df,
    get_taxonomy_by_name,
    get_train_eval_test,
)
from .generate import generate_base, repr_generation_diff
from .main import compute_patterns_main, get_current_config_hash_main
from .models import get_default_tokenizer, get_model, get_tokenizer
from .repr_patterns import (
    get_patterns_for_repr,
    plot_attention_pattern,
    plot_multisentence_multilayer,
    plot_singlesentence_multihead,
)
from .titles_utils import get_titles, get_titles_no_others, parse_single_title
