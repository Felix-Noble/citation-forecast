#!/usr/bin/env python3
# pyright: standard
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportMissingTypeStubs=false
#from config import config
#from src.utils.logging import setup_logger
import polars as pl
from transformers import AutoTokenizer
import os
from logging import getLogger
from pathlib import Path
#logger = getLogger(Path(__file__).stem)
#_ = setup_logger(logger, config.logging)

os.environ['POLARS_MAX_THREADS'] = '16'

def main(
        lf: pl.LazyFrame,
        tokeniser_path: str,
        columns: list[str],
):
    tokeniser = AutoTokenizer.from_pretrained(
        tokeniser_path,
        use_fast=True,
    )
    def tokenise_partition(text: pl.Series):
        tokens = tokeniser(
            (tokeniser.bos_token + text + tokeniser.eos_token).to_list(), 
            return_tensors=None,
            truncation=False,
            padding=False,
        )['input_ids']

        return pl.Series(
            name=text.name + '_tokens',
            values=tokens, 
            dtype=pl.List(pl.Int32),
        )

    # TODO make this relative to project package root
    if not os.path.exists(f'./{tokeniser}'):
        tokeniser.save_pretrained(tokeniser_path)

    for col in columns:
        lf = lf.with_columns(
            pl.col(col)
            .fill_null('')
            .map_batches(tokenise_partition)
            .alias(f'{col}_tokens')
        )
    return lf
