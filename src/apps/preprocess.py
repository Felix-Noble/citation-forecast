import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["POLARS_MAX_THREADS"] = "32"
from config import config
from src.data.preprocess import clean_step, tokenise_step
from src.utils.logging import setup_logger

from rich.progress import Progress, TaskID, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from datetime import datetime
import math
import json
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

def measure_lf(lf: pl.LazyFrame) -> int:
    return lf.select(pl.len()).collect(engine='streaming').item()

def export_parquet(
    lf: pl.LazyFrame,
    destination: Path,
    n_partitions: int,
    progress_bar: Progress | None=None,
    progress_task: TaskID | None=None,
    compression_level: int = 1,
) -> None:

    n_rows = measure_lf(lf)
    rows_per_part = math.ceil( n_rows / n_partitions )

    lf = lf.with_row_index("idx").with_columns(
    (pl.col("idx") // rows_per_part).clip(0, n_partitions - 1).alias("part")
    )
    # 3. Sink each part
    for i in range(n_partitions):
        (
            lf.filter(pl.col("part") == i)
                .drop(["idx", "part"])
                .sink_parquet(
                    destination / f"part_{i}.parquet",
                    statistics=True,
                    compression='zstd',
                    compression_level=compression_level
                )
        ) 
        if progress_bar is not None and progress_task is not None:
            progress_bar.update(progress_task, advance=rows_per_part)

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
    clean_cols: list[str] = typer.Option(
            [],
            '--clean-col', '-c',
            help = 'Data columsn to clean, must be str, pass multiple flags for multiple cols'
            ),
    clean_levels: list[int] = typer.Option(
            [],
            '--clean-level', '-cl',
            help='Level to clean at column at, must be given in the same order as clean cols'
            ),
    clean_min_len: list[int] = typer.Option(
            [],
            '--clean-min-len', '-cml',
            help='Minimum length for the column being cleaned, must be given in the same order as clean cols'
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
    n_partitions: int = typer.Option(
            64,
            '--partitions', '-p',
            help = 'N. partitions to split the dataset into'
            ),
    compression_level: int = typer.Option(
            1,
            '--compression-level', 
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
    clear_temp: bool = typer.Option(
            True,
            '--clear-temp',
            help='Delete temp files, whether successful or not'
            ),
    ctx: typer.Context = None,
    ) -> None:
    # Safety Checks
    assert (tokeniser and tokenise_cols) or (not tokeniser and not tokenise_cols), 'Provide columns to be tokenised'
    assert (embed_model and embed_cols) or (not embed_model and not embed_cols), 'Provide columns to be embedded'
    assert (len(clean_cols) == len(clean_levels) == len(clean_min_len)), 'Clean cols, levels, and min-lens must be givein in same quanity'

    try:
        # Setup
        if not name:
            destination = Path(origin).parent / f'{str(Path(origin).stem)}_preprocessed'
        else:
            destination = Path(origin).parent / name
        if os.path.exists(destination):
            assert (len(os.listdir(destination)) < 1), "Ensure destination is empty"

        n_rows: int = pl.scan_parquet(origin, extra_columns='ignore').select(pl.len()).collect(engine='streaming').item()
        alternator = value_alternator()

        progress_bar = Progress(
            TextColumn('[bold blue] {task.description}', justify='left'),
            BarColumn(bar_width=40),
            TextColumn('[task.completed]{task.completed}/{task.total}'),
            TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
            TimeElapsedColumn(),
            TextColumn('<'), 
            TimeRemainingColumn(),
            speed_estimate_period=60.0 * 10, # mins
        )     
        progress_bar.start()

        # Main Logic
        lf: pl.LazyFrame = pl.scan_parquet(origin, extra_columns='ignore')
        if dry_run:
            lf = lf.slice(0,500)
        
        # License filtering
        l1 = measure_lf(lf)
        lf = lf.filter(pl.col('is_license_safe'))
        l2 = measure_lf(lf)
        logger.info(f'Dropped {l1 - l2:,} with non-permissive licenses') 

        if clean_cols:
            for col, level, min_len in zip(clean_cols, clean_levels, clean_min_len):
                l1 = measure_lf(lf)
                lf = clean_step(
                    lf=lf,
                    col=col,
                    min_len=min_len,
                    level=level,
                )
                l2 = measure_lf(lf)
                logger.info(f'Dropped {l1 - l2:,} in {col} at clean lvl {level}') 
        else:
            logger.warning('No cols selected for cleaning')

        if drop_na_cols:
            for col in drop_na_cols:
                l1 = measure_lf(lf)
                lf = lf.drop_nulls(subset=col)
                l2 = measure_lf(lf)
                logger.info(f"Dropped {l1 - l2:,} nulls in {col}") 
        else:
            logger.warning('No columns set to drop nulls')

        if start_date is not None:
            l1 = measure_lf(lf)
            lf = lf.filter((pl.col('publication_date') >= start_date))
            l2 = measure_lf(lf)
            logger.info(f'Dropped {l1 - l2:,} prior to {start_date}') 
        else:
            logger.warning('No start date set')

        if end_date is not None:
            l1 = measure_lf(lf)
            lf = lf.filter((pl.col('publication_date') < end_date))
            l2 = measure_lf(lf)
            logger.info(f'Dropped {l1 - l2:,} after {end_date}')
        else:
            logger.warning('No end date set')

        if field_id:
            l1 = measure_lf(lf)
            lf = lf.filter(pl.col('field_id').is_in(field_id))
            l2 = measure_lf(lf)
            logger.info(f'Dropped {l1 - l2:,} field id not in {field_id}') 
        else:
            logger.warning('No field id filter set')

        if languages:
            l1 = measure_lf(lf)
            lf = lf.filter(pl.col('language').is_in(languages))
            l2 = measure_lf(lf)
            logger.info(f'Dropped {l1 - l2:,} lang not in {languages}') 
        else:
            logger.warning('No Language filters set')

        if types:
            l1 = measure_lf(lf)
            lf = lf.filter(pl.col('type').is_in(types))
            l2 = measure_lf(lf)
            logger.info(f'Dropped {l1 - l2:,} type not in {types}') 
        else:
            logger.warning('No document type filters set')

        if embed_model:
            logger.warning('Embed feature is a work in progress, leave out for now')

        # Checkpoint
        n_rows = measure_lf(lf)
        progress = progress_bar.add_task('Clean/Filter', total=n_rows)

        temp_loc = Path(f'./temp{next(alternator)}')
        os.makedirs(temp_loc)
        export_parquet(
            lf=lf,
            destination=temp_loc,
            n_partitions=n_partitions,
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
        metadata = { k: str(v) for k,v in ctx.params.items() }
        with open( destination / 'metadata.json', 'w' ) as f:
            json.dump(metadata, f)

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
                n_partitions=n_partitions,
                progress_bar=progress_bar,
                progress_task=progress,
                compression_level=compression_level,
            )
    
    except Exception as e:
        try:
            logger.error(e)
        except:
            logger.error('Failed to log error')
        finally:
            raise e

    finally: 
        if clear_temp:
            for i in range(2):
                logger.info(f'Deleting temp{i}')
                if os.path.exists(f'./temp{i}'):
                    shutil.rmtree(f'./temp{i}')

