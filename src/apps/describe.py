from config import Config, TrainConfig, config, env
from src.utils.logging import setup_logger
from src.builders import \
        build_datasets, \
        build_eval_example_progress, \
        build_epoch_progress, \
        build_dataloader, \
        build_eval_tracker, \
        build_model, \
        build_loss \

from src.data.datasets import BinaryCategoricalDataset, OrdinalDataset
from src.training.eval import eval_model
from src.training.tracking import MetricTracker, ClassificationTracker, log_params, log_lrs
import polars as pl
import logging
from logging import getLogger
import copy
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
from typing import NamedTuple
import typer 
import os
import sys

class DateTimeVals(NamedTuple):
    year: int 
    month: int 
    day: int 

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

app = typer.Typer(pretty_exceptions_enable=False)

@app.callback(invoke_without_command=True)

def main(
        dataset: str = typer.Option(
            '',
            '--dataset', '-d',
            help='Dataset name'
            ),
        start_time: datetime | None = typer.Option(
            None,
            '--start-time', '-s',
            help='Start date/time to filter by, inlcusive'
            ),
        end_time: datetime | None = typer.Option(
            None,
            '--end-time', '-e',
            help='End date/time to filter by, exclusive'
            ),
        time_col: str = typer.Option(
            'publication_date',
            '--time-col', '-tc',
            help='Date/Time col in dataset.'
            ),
        describe_cols: list[str] = typer.Option(
            [],
            '--column', '-c',
            help='Columns to describe'
            ),
        buckets: list[float] = typer.Option(
            [],
            '--bucket', '-b',
            help='Borders of buckets to describe count by, creates list of inclusive upper limits'
            )
):
    lf = pl.scan_parquet(list((env.STAGED_LOC / dataset).glob('*.par*')))
    
    if start_time is not None:
        lf = lf.filter(pl.col(time_col) >= start_time)
    if end_time is not None:
        lf = lf.filter(pl.col(time_col) < end_time)
    
    n_buckets = len(buckets)
    for col in describe_cols:
        bucket_counts = { f'{buckets[i]}-{buckets[i+1]}': None for i in range(n_buckets - 1) }
        weights = { k: v for k,v in bucket_counts.items() }

        for i in range(n_buckets - 1):
            bucket_counts[f'{buckets[i]}-{buckets[i+1]}'] = (
                    lf.filter((pl.col(col) >= buckets[i]) & (pl.col(col) < buckets[i+1]) )
                    .select(pl.len())
                    .collect(engine='streaming')
                    .item()
                    )
        total = sum([v for v in bucket_counts.values()])
        for k in weights.keys():
            weights[k] = total/((n_buckets - 1) * bucket_counts[k])
         
        print()
        logger.info(f'Describing {col}')
        logger.info('counts: ')

        logger.info(bucket_counts)
        logger.info('weights: ')
        logger.info(weights)

        description = lf.select(pl.col(col)).describe()
        logger.info(description)

    logger.info('Finished Describe')

