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
    import json
    import ast
    import dask
    dask.config.set({"dataframe.convert-string": False})
    import dask.dataframe as dd
    import pandas as pd
    from dask.distributed import Client, progress
    import pyarrow as pa
    from transformers import AutoTokenizer
    from rich.console import Console

    console = Console()

    def tokenize_partition(text: pd.Series, tokenizer):
        tokens = tokenizer(
            (tokenizer.bos_token + text + tokenizer.eos_token).to_list(), 
            return_tensors=None,
            truncation=False,
            padding=False,
        )['input_ids']

        return pd.Series(
            data=tokens, 
            index=text.index,
            dtype=object,
        )

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
    if dry_run:
        ddf = ddf.sample(frac=0.001).repartition(npartitions=4)
    logger.info(f'Loaded {int(len(ddf)):,} rows')
    ddf = ddf.dropna(subset=columns)
    logger.info(f'Dropped nulls in [{", ".join(columns)}], {int(len(ddf)):,} remaining')
    ddf = ddf.reset_index(drop=True) 
    for col in columns:
        meta = pd.Series([], dtype='object', name=f'{col}_tokens') 
        ddf[f'{col}_tokens'] = ddf[col].map_partitions(
            tokenize_partition,
            tokenizer=tokenizer,
            meta=meta,
        )
        ddf[f'{col}_tokens'] = ddf[f'{col}_tokens'].apply(
            lambda val : val if isinstance(val, list) else ast.literal_eval(val),
            meta=meta,
        )

    logger.info(f'Tokenising [ [green]{", ".join(columns)}[/green] ]')
    ddf = ddf.persist()
    progress(ddf)

    schema = {}
    for col in columns:
        schema[f'{col}_tokens'] = pa.list_(pa.int64())

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
            schema=schema,
        )
        progress(client.compute(ddf_out))
        with open(Path(data_path) / 'metadata.json', 'w') as f:
            json.dump({
                'tokenizer': tokenizer_path
            }, f)

        console.print('[green]Finished [/green]')
    else:
        console.print(f'Prepared schema: {schema}')
        test_out = './testing/tokenise-dry-run-dir/'
        console.print(f'[red]Dry run sample saving to[/red] [yellow]{test_out}[/yellow]')
        ddf_out = dd.to_parquet(
            ddf,
            test_out,
            compression='zstd',
            compression_level=1,
            write_statistics=True,
            overwrite=True,
            compute=False,
            schema=schema,
        )
        progress(client.compute(ddf_out))
        console.print('[red]Dry Run complete[/red]')

    client.close()

if __name__ == '__main__':
    app()
