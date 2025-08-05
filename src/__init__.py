# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import os

from dotenv import load_dotenv

# Load env
load_dotenv()

# HF Home?
if not os.path.isdir(os.getenv("HF_HOME")):
    os.environ["HF_HOME"] = os.getenv("HF_HOME_BIS")

# Additional imports
# isort:skip
from loguru import logger

from .utils import paths

logger.add(
    paths.logs_path / "main.log",
    rotation="10 MB",
)
