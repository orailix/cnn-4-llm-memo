# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import pathlib

# Default root
root = pathlib.Path(__file__).parent.parent.parent

# Create `.cache`  and `logs` dirs
cache_path = root / ".cache"
cache_path.mkdir(exist_ok=True, parents=False)
logs_path = root / "logs"
logs_path.mkdir(exist_ok=True, parents=False)
output = root / "output"
output.mkdir(exist_ok=True, parents=False)
figures = root / "figures"
figures.mkdir(exist_ok=True, parents=True)
