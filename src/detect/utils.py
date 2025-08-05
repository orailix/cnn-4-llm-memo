# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import torch


def check_model_parameter_sizes(model: torch.nn.Module):

    # Number of groups of parameters
    print("Number of groups of parameters {}".format(len(list(model.parameters()))))
    print("-" * 55)
    # Print parameters

    total = 0
    for name, param in model.named_parameters():
        print(f"{name:17} : [{param.numel():6}] {param.size()}")
        total += param.numel()

    print(f"{'TOTAL':17} : [{total:6}]")

    print("-" * 55)
