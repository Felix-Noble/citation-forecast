import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["POLARS_MAX_THREADS"] = "16"
from config import config
from src.data.preprocess import clean_step, tokenise_step
from src.utils.logging import setup_logger

from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from datetime import datetime
import typer 
import shutil
from pathlib import Path
import polars as pl
from logging import getLogger

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)
app = typer.Typer(pretty_exceptions_enable=False)

def value_alternator(n: int = 2):
    i = 0 
    while True:
        yield i
        i += 1
        if i >= n:
            i = 0

def export_parquet(
    lf: pl.LazyFrame,
    destination: str | Path,
    rows_per_file: int,
    progress_bar: Progress | None=None,
    progress_task: TaskID | None=None,
    compression_level: int = 4,
                    ) -> None:
    i = 0
    while True:
        lf_slice = lf.slice(i*rows_per_file, rows_per_file)
        rows = lf_slice.select(pl.len()).collect(engine='streaming').item()
        if rows < 1:
            break
        lf_slice.sink_parquet(
            f'{destination}/part{i}.parquet',
            statistics=True,
            compression='zstd',
            compression_level=compression_level
        ) 
        i += 1 
        if progress_bar is not None and progress_task is not None:
            progress_bar.update(progress_task, advance=rows)

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
        300_000,
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
            '--start-date', '-sd',
            help = 'Start date to filter by, inclusive'
            ),
    end_date: datetime | None = typer.Option(
            None,
            '--end-date', '-ed',
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
    if os.path.exists(destination):
        assert (len(os.listdir(destination)) < 1), "Ensure destination is empty"
    n_rows: int = pl.scan_parquet(origin).select(pl.len()).collect(engine='streaming').item()
    alternator = value_alternator()

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
    else:
        logger.warning('No field id filter set')
    if drop_na_cols:
        lf = lf.drop_nulls(subset=drop_na_cols)
    else:
        logger.warning('No columns set to drop nulls')
    if languages:
        lf = lf.filter(pl.col('language').is_in(languages))
    else:
        logger.warning('No Language filters set')
    if types:
        lf = lf.filter(pl.col('type').is_in(types))
    else:
        logger.warning('No document type filters set')
    if clean:
        lf = clean_step(
                lf=lf,
                columns=clean_cols
                        )
    if embed_model:
        logger.warning('Embed feature is a work in progress, leave out for now')

    # finish previous steps before continuing
    progress = progress_bar.add_task('Clean/Filter', total=n_rows)
    temp_loc = f'./temp{next(alternator)}'
    os.makedirs(temp_loc)
    export_parquet(
        lf=lf,
        destination=temp_loc,
        rows_per_file=rows_per_file,
        progress_bar=progress_bar,
        progress_task=progress,
        compression_level=4,
    )
    lf = pl.scan_parquet(temp_loc)
    n_rows: int = pl.scan_parquet(temp_loc).select(pl.len()).collect(engine='streaming').item()
    # Export 
    if dry_run:
        print(lf.collect())
    progress = progress_bar.add_task('Rows', total=n_rows)
    os.makedirs(destination, exist_ok=True)
    if tokeniser:
        for file in Path(temp_loc).glob('*.par*'):
            lf_file = pl.scan_parquet(file)
            rows = lf_file.select(pl.len()).collect().item()
            lf_file = tokenise_step(
                lf=lf_file,
                tokeniser_path=tokeniser,
                columns=tokenise_cols,
            )
            lf_file.sink_parquet(
                destination / f'{file.stem}{file.suffix}',
                compression='zstd',
                compression_level=compression_level,
            )
            progress_bar.update(progress, advance=rows)

    else:
        export_parquet(
            lf=lf,
            destination=destination,
            rows_per_file=rows_per_file,
            progress_bar=progress_bar,
            progress_task=progress,
            compression_level=compression_level,
        )

    for i in range(2):
        if os.path.exists(f'./temp{i}'):
            shutil.rmtree(f'./temp{i}')

