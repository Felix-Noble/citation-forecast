# pyright: basic
# pyright: reportAttributeAccessIssue=false, reportPrivateImportUsage=false
import typer
app = typer.Typer()

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

        return pd.Series(tokens)

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
    )
    ddf = ddf.dropna(subset=columns)
    for col in columns:
        ddf[col] = tokenizer.bos_token + ddf[col] + tokenizer.eos_token 
        console.print(f'Tokenising [green]{col}[/green]')
        ddf[f'{col}_tokens'] = ddf[col].map_partitions(
                                            tokenize_partition, 
                                            tokenizer=tokenizer,
                                            meta=(None, 'object')
                                        )

    ddf = ddf.persist()
    progress(ddf)
    
    if not dry_run: 
        console.print('[yellow] Writing output[/yellow]')
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
        console.print('[green] Finished [/green]')
    else:
        console.print('[red] Dry Run complete [/red]')

def run_app():
    app()

if __name__ == '__main__':
    app()
