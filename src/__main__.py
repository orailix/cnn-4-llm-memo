# `cnn_4_llm_memo`

# Copyright 2025-present Laboratoire d'Informatique de Polytechnique.
# Apache Licence v2.0.

import typing as t

import typer

from .detect import train_cnn_2d
from .patterns import compute_patterns_main, get_current_config_hash_main

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def compute_patterns():
    compute_patterns_main()


@app.command()
def get_current_config_hash():
    get_current_config_hash_main()


@app.command()
def train(config: t.Optional[str] = None):
    train_cnn_2d(config)


if __name__ == "__main__":
    app()
