from src.data.preprocess import clean_step, tokenise_step
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from datetime import datetime
import typer 
import os
from pathlib import Path
import polars as pl

app = typer.Typer(pretty_exceptions_enable=False)

@app.callback(invoke_without_command=True)
def main(
    origin: str = typer.Argument(
        help = 'path to data',
        ),
    name: str = typer.Option(
        '',
        '--name', '-n',
        help = 'Name of exported dataset, defaults to "origin"_prepped'
        ),
    clean: bool = typer.Option(
        False,
        '--clean','-c',
        help = 'clean string data signified in "clean-cols" arg',
        ),
    # list of strings
    clean_cols: list[str] = typer.Option(
        [],
        '--clean-col', '-cc',
        help = 'Data columsn to clean, must be str, pass multiple flags for multiple cols'
        ),
    tokeniser: str = typer.Option(
        '',
        '--tokeniser', '-t',
        help = 'Which tokeniser to use when tokenise (-t) flag given',
        ),
    tokenise_cols: list[str] = typer.Option(
        [],
        '--tokenise-col', '-tc',
        help = 'Data columsn to tokenise, must be str, pass multiple flags for multiple cols'
        ),
    embed_model: str = typer.Option(
            '',
            '--embed-model', '-em',
            help = 'Model to embed string data with, inactive if empty'
            ),
    embed_cols: list[str] = typer.Option(
            [],
            '--embed-cols', '-ec',
        help = 'Data columsn to tokenise, must be str, pass multiple flags for multiple cols'
            ),
    rows_per_file: int = typer.Option(
        1_000_000,
        '--rows-per-file',
        help = 'Max rows per file'
        ),
    compression_level: int = typer.Option(
        4,
        '--compression-level', '-cl',
        help = 'zstd compression level'
        ),
    start_date: datetime | None = typer.Option(
            None,
            '--start-date',
            help = 'Start date to filter by, inclusive'
            ),
    end_date: datetime | None = typer.Option(
            None,
            '--end-date',
            help = 'End date to filter by, exclusive'
            ),
    field_id: list[int] = typer.Option(
            [],
            '--field-id', '-fid',
            help = 'Field id/s to include'
            ),
    drop_na_cols: list[str] = typer.Option(
            [],
            '--drop-na-col', 
            help = 'Cols to drop nulls in'
            ),
    languages: list[str] = typer.Option(
            ['en'],
            '--lang',
            help = 'Languages to include'
            ),
    types: list[str] = typer.Option(
            [],
            '--type',
            help = 'Types of document (eg. article, book chapter) to include'
            ),
    dry_run: bool = typer.Option(
            False,
            '--dry-run',
            help = 'Test/Dry run with 500 sample slice of data',
            ),
    calc_total_rows: bool = typer.Option(
            False,
            '--calc-total-rows',
            help = 'Calculate total rows for progress bar, requires whole frame to memory'
            )
    ) -> None:
    # Safety Checks
    assert (clean and clean_cols) or (not clean and not clean_cols), 'Provide columns to be cleaned'
    assert (tokeniser and tokenise_cols) or (not tokeniser and not tokenise_cols), 'Provide columns to be tokenised'
    assert (embed_model and embed_cols) or (not embed_model and not embed_cols), 'Provide columns to be embedded'

    # Setup
    if not name:
        destination = Path(origin).parent / f'{str(Path(origin).stem)}_preprocessed'
    else:
        destination = Path(origin).parent / name
    os.makedirs(destination)

    progress_bar = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TextColumn('<'), 
        TimeRemainingColumn(),
        #speed_estimate_period=60.0 * 10, # mins
    )     
    progress_bar.start()

    # Main Logic
    lf: pl.LazyFrame = pl.scan_parquet(origin)
    if dry_run:
        lf = lf.slice(0,500)

    if start_date is not None:
        lf = lf.filter((pl.col('publication_date_int') >= start_date.toordinal()))
    if end_date is not None:
        lf = lf.filter((pl.col('publication_date_int') < end_date.toordinal()))
    if field_id:
        lf = lf.filter(pl.col('field_id').is_in(field_id))
    if drop_na_cols:
        lf = lf.drop_nulls(subset=drop_na_cols)
    if languages:
        lf = lf.filter(pl.col('language').is_in(languages))
    if types:
        lf = lf.filter(pl.col('type').is_in(types))
    if clean:
        lf = clean_step(
                lf=lf,
                columns=clean_cols
                        )
    if tokeniser:
        lf = tokenise_step(
                lf=lf,
                tokeniser_path=tokeniser,
                columns=tokenise_cols,
                           )
    if embed_model:
        raise ValueError('Feature is a work in progress, leave out for now')

    # Export 
    if dry_run:
        print(lf.collect())
    n_rows = -1
    if calc_total_rows:
        n_rows: int = lf.select(pl.len()).collect(engine='streaming').item()
    progress = progress_bar.add_task('Rows', total=n_rows)

    i = 0
    while True:
        lf_slice = lf.slice(i*rows_per_file, (i+1)*rows_per_file)
        len: int = lf_slice.select(pl.len()).collect(engine='streaming').item()
        if len < 1:
            break
        lf_slice.sink_parquet(
            f'{destination}/part{i}.parquet',
            statistics=True,
            compression='zstd',
            compression_level=compression_level
        ) 
        i += 1 
        progress_bar.update(progress, advance=len)
  
