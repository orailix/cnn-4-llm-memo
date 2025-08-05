# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import h5py
import numpy as np
import torch
from accelerate import Accelerator
from datasets import Dataset
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import PatternsConfig
from .datasets import get_train_eval_test
from .models import get_model


def get_current_config_hash_main():
    cfg = PatternsConfig.from_env()
    print(cfg.get_id())


def compute_patterns_main():

    # Config
    cfg = PatternsConfig.from_env()

    # Logger
    logger.info(f"Starting patterns computation with config: {cfg}")

    # Data
    logger.info(f"Loading dataset")
    train_set = get_train_eval_test(cfg)

    # Model
    logger.info(f"Loading model")
    model = get_model(cfg, hooked=True, eager_attention=True)

    # Inference parameters
    batch_size = cfg.inference_bs
    output_file = cfg.get_patterns_path()
    output_file.parent.mkdir(exist_ok=True, parents=True)
    num_layers = len(model.blocks)

    # Dataset
    logger.debug(
        f"Filtering already computed samples and duplicates. Len before filtering: {len(train_set)}"
    )
    filtered_train_set = train_set.drop_duplicates(subset="sequence_id")
    hf_dataset = Dataset.from_pandas(filtered_train_set[["sequence_id", "tokens"]])
    logger.debug(f"End of filtering. Len after filtering: {len(filtered_train_set)}")

    # Collate
    def collate_fn(batch):
        sequence_ids = [item["sequence_id"] for item in batch]
        tokens = torch.stack(
            [torch.tensor(item["tokens"], dtype=torch.int) for item in batch]
        )
        return sequence_ids, tokens

    # Data Loader
    data_loader = DataLoader(hf_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Initialize HDF5 File
    logger.info(f"Starting inference")
    with h5py.File(output_file, "w") as h5f:
        # Create a dataset for sequence_ids

        # Iterating through the DataLoader
        for sequence_ids, batch in tqdm(data_loader):
            # Move tokens to the appropriate device
            batch = batch.to(device)

            # Run the model with cache
            _, cache = model.run_with_cache(batch)

            # Extract attention patterns
            attention_patterns = torch.stack(
                [cache["pattern", layer, "attn"] for layer in range(num_layers)],
                dim=1,
            ).cpu()

            # Store tensors in HDF5
            for seq_id, tensor in zip(sequence_ids, attention_patterns):
                # Store each tensor under its sequence_id
                h5f.create_dataset(
                    str(seq_id),
                    data=tensor.numpy().astype(np.float16),
                )

    # Logging
    logger.info(f"End of computation!")
