#!/usr/bin/env python3
# pyright: standard
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportMissingTypeStubs=false
# from utils.logging import setup_logger
import os
from logging import getLogger
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer

# logger = getLogger(Path(__file__).stem)
# _ = setup_logger(logger, config.logging)
tokeniser = None


def main(
    lf: pl.LazyFrame,
    tokeniser_path: str,
    columns: list[str],
):

    # TODO make this relative to project package root
    if not os.path.exists(f"./{tokeniser_path}"):
        tokeniser = AutoTokenizer.from_pretrained(
            tokeniser_path,
            use_fast=False,
        )
        tokeniser.save_pretrained(tokeniser_path)
        tokeniser = None

    def tokenise_partition(text: pl.Series):
        global tokeniser
        if tokeniser is None:
            tokeniser = AutoTokenizer.from_pretrained(
                tokeniser_path,
                add_eos_token=True,
                add_bos_token=True,
                use_fast=False,
            )
        tokens = tokeniser(
            text.to_list(),
            add_special_tokens=True,
            return_tensors=None,
            truncation=False,
            padding=False,
        )["input_ids"]

        return pl.Series(
            name=text.name + "_tokens",
            values=tokens,
            dtype=pl.List(pl.Int64),
        )

    for col in columns:
        lf = lf.with_columns(
            pl.col(col)
            .fill_null("")
            .map_batches(tokenise_partition, return_dtype=pl.List(pl.Int64))
            .alias(f"{col}_tokens")
        )
        lf = lf.with_columns(
            pl.when(pl.col(col).is_null())
            .then(pl.lit([], dtype=pl.List(pl.Int64)))
            .otherwise(pl.col(f"{col}_tokens"))
            .alias(f"{col}_tokens")
        )
    return lf
