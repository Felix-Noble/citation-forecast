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
    data_path: str,
    tokenizer_path: str,
    columns: list[str],
    dry_run: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
    )
    def tokenize_partition(text: pl.Series):
        tokens = tokenizer(
            (tokenizer.bos_token + text + tokenizer.eos_token).to_list(), 
            return_tensors=None,
            truncation=False,
            padding=False,
        )['input_ids']

        return pl.Series(
            name=text.name + '_tokens',
            values=tokens, 
            dtype=pl.List(pl.Int32),
        )

    if not os.path.exists(f'./{tokenizer}'):
        tokenizer.save_pretrained(tokenizer_path)
    lf = pl.scan_parquet(data_path)

    if dry_run:
        lf = lf.slice(0,500)

    for col in columns:
        lf = lf.with_columns(
            pl.col(col)
            .fill_null('')
            .map_batches(tokenize_partition)
            .alias(f'{col}_tokens')
        )

    if not dry_run: 
        i = 0
        n = 1_000_000
        while True:
            lf_slice = lf.slice(i*n, n)
            len = lf_slice.select(pl.len()).collect(engine='streaming').item()
            if len < 1:
                break
            print(f'writing part {i} | n={len}')
            lf_slice.sink_parquet(
                f'{data_path}_tokenised/part{i}.parquet',
                statistics=True,
                compression='zstd',
                compression_level=4
            ) 
            i += 1
    else:
        len = lf.select(pl.len()).collect()
        print(f'{len} rows tokenised')
        print(lf.columns)
        print(lf.collect())

if __name__ == '__main__':
    main(
        data_path='/home/fnoble/data/staged/all_clean',
        columns=['title', 'abstract'],
        tokenizer_path='openai/gpt-oss-120b',
        dry_run=False,
    )
