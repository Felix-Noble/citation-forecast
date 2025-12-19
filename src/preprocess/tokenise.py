#!/usr/bin/env python3
# pyright: standard
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportMissingTypeStubs=false
import typer
from config.config import config
from src.utils.logging import setup_logger
from logging import getLogger
from pathlib import Path
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)
app = typer.Typer()

@app.callback(invoke_without_command=True)
@app.command()
def main(
    data_path: str,
    tokenizer_path: str,
    columns: list[str] = typer.Option(..., '--column', '-c'),
    n_workers: int = 4,
    threads_per_worker: int = 8,
    mem_limit: int = 28,
    dry_run: bool = False,
):
    import os
    import dask.dataframe as dd
    import pandas as pd
    from dask.distributed import Client, progress
    from transformers import AutoTokenizer
    from rich.console import Console

    console = Console()

    def tokenize_partition(text: pd.Series, tokenizer):
        tokens = tokenizer(
            text.to_list(), 
            return_tensors=None,
            truncation=False,
            padding=False,
        )['input_ids']

        return pd.Series(tokens, text.index)

    client = Client( 
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=f'{mem_limit}GB'
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        use_fast=True,
    )

    if not os.path.exists(f'./{tokenizer}'):
        tokenizer.save_pretrained(tokenizer_path)
    ddf = dd.read_parquet(
        data_path,
        engine='fastparquet'
    ).persist()

    logger.info(f'Loaded {int(len(ddf)):,} rows')
    ddf = ddf.dropna(subset=columns)
    logger.info(f'Dropped nulls in [{", ".join(columns)}], {int(len(ddf)):,} remaining')
    ddf = ddf.compute().reset_index(drop=True, inplace=False) 
    for col in columns:
        #meta = pd.Series([], dtype='object', name=f'{col}_tokens') 
        ddf[col] = tokenizer.bos_token + ddf[col] + tokenizer.eos_token 
        console.print(f'Tokenising [green]{col}[/green]')
        ddf[f'{col}_tokens'] = tokenize_partition(
            ddf[col],
            tokenizer,
        )

    ddf = dd.from_pandas(ddf, npartitions=64)
    ddf = ddf.persist()
    progress(ddf)

    if not dry_run: 
        console.print('[yellow]Writing output[/yellow]')
        ddf_out = dd.to_parquet(
            ddf,
            data_path,
            compression='zstd',
            compression_level=1,
            write_statistics=True,
            overwrite=True,
            compute=False,
        )
        progress(client.compute(ddf_out))
        console.print('[green]Finished [/green]')
    else:
        console.print('[red]Dry Run complete [/red]')

if __name__ == '__main__':
    app()
