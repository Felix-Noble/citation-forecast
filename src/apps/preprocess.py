import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
import json
import math
import gc
from datetime import datetime
from logging import getLogger
from pathlib import Path

import typer
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from data.preprocess import clean_step, tokenise_step
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)
app = typer.Typer(pretty_exceptions_enable=False)

from transformers import AutoModel, AutoTokenizer
import polars as pl
import torch

MODEL_NAME = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# 2. LOAD DIRECTLY IN BFLOAT16 (Crucial Step!)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,  # Force HF to download/load in BF16 directly
    attn_implementation="sdpa"   # Keep fast attention active
)
if not os.path.exists(MODEL_NAME):
    tokenizer.save_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_NAME)

bos_token = tokenizer.cls_token
eos_token = tokenizer.sep_token

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Device: {device}')
model = model.to(device).eval()
model.compile(mode='default')

def embed_batch(series: pl.Series, *args, **kwargs) -> pl.Series:
    """Helper function to process a batch of strings and return embeddings."""
    # Convert the Polars Series batch to a Python list of strings
    texts = series.to_list()

    # Tokenize inputs
    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    attention_mask = inputs["attention_mask"]      # Shape: [Batch, Seq_Len]
        # Generate embeddings without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

        # 1. Extract the token embeddings and the attention mask
        token_embeddings = outputs.last_hidden_state  # Shape: [Batch, Seq_Len, Hidden_Dim]

        # 2. Expand the 2D mask to a 3D mask to match the embeddings
        # Becomes shape: [Batch, Seq_Len, Hidden_Dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        # 3. Zero out the padding tokens and sum up the valid token vectors
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)

        # 4. Count the number of non-padded tokens (clamp to 1e-9 to prevent division-by-zero)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

        # 5. Calculate the true, uncorrupted mean
        embeddings = sum_embeddings / sum_mask

    # Convert PyTorch tensor back to a list of lists (floats) for Polars
    embeddings_list = embeddings.float().cpu().tolist()

    torch.cuda.empty_cache()

    # Return as a Polars Array type (highly optimized for fixed-size vectors)
    return pl.Series(embeddings_list, dtype=pl.Array(pl.Float32, width=model.config.hidden_size))

@app.callback(invoke_without_command=True)
def main(
    origin: Path = typer.Argument(
        help="path to data",
    ),
    name: str = typer.Option(
        "",
        "--name",
        "-n",
        help='Name of exported dataset, defaults to "origin"_prepped',
    ),
    clean_cols: list[str] = typer.Option(
        [],
        "--clean-col",
        "-c",
        help="Data columsn to clean, must be str, pass multiple flags for multiple cols",
    ),
    clean_levels: list[int] = typer.Option(
        [],
        "--clean-level",
        "-cl",
        help="Level to clean at column at, must be given in the same order as clean cols",
    ),
    clean_min_len: list[int] = typer.Option(
        [],
        "--clean-min-len",
        "-cml",
        help="Minimum length for the column being cleaned, must be given in the same order as clean cols",
    ),
    tokeniser: str = typer.Option(
        "",
        "--tokeniser",
        "-t",
        help="Which tokeniser to use when tokenise (-t) flag given",
    ),
    tokenise_cols: list[str] = typer.Option(
        [],
        "--tokenise-col",
        "-tc",
        help="Data columsn to tokenise, must be str, pass multiple flags for multiple cols",
    ),
    embed_model: str = typer.Option(
        "",
        "--embed-model",
        "-em",
        help="Model to embed string data with, inactive if empty",
    ),
    embed_cols: list[str] = typer.Option(
        [],
        "--embed-cols",
        "-ec",
        help="Data columsn to tokenise, must be str, pass multiple flags for multiple cols",
    ),
    n_partitions: int = typer.Option(
        0,"--partitions", "-p", help="N. partitions to split the dataset into"
    ),
    rows_per_part: int = typer.Option(
        0,'--rows-per-part','-rp', help='Rows per partition'
    ),
    compression_level: int = typer.Option(
        1, "--compression-level", help="zstd compression level"
    ),
    start_date: datetime | None = typer.Option(
        None, "--start-date", "-sd", help="Start date to filter by, inclusive"
    ),
    end_date: datetime | None = typer.Option(
        None, "--end-date", "-ed", help="End date to filter by, exclusive"
    ),
    field_id: list[int] = typer.Option(
        [], "--field-id", "-fid", help="Field id/s to include"
    ),
    drop_na_cols: list[str] = typer.Option(
        [], "--drop-na-col", help="Cols to drop nulls in"
    ),
    languages: list[str] = typer.Option([], "--lang", help="Languages to include"),
    types: list[str] = typer.Option(
        [], "--type", help="Types of document (eg. article, book chapter) to include"
    ),
    filt_license: bool = typer.Option(True, help="Remove non-permissive licenses"),
    replace_non_permissive_cols: list[str] = typer.Option(
        [],
        "--replace-non-permissive-col",
        help=f"Replace non-permissively licensed col with {None}",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Test/Dry run with 500 sample slice of data",
    ),
    max_threads: int = typer.Option(
        8, "--max-threads", help="Max threads to use, e.g. polars env variable"
    ),
    clear_temp: bool = typer.Option(
        True, "--clear-temp", help="Delete temp files, whether successful or not"
    ),
    ctx: typer.Context = None,
) -> None:
    # Safety Checks
    assert (tokeniser and tokenise_cols) or (not tokeniser and not tokenise_cols), (
        "Provide columns to be tokenised"
    )
    assert (embed_model and embed_cols) or (not embed_model and not embed_cols), (
        "Provide columns to be embedded"
    )
    assert len(clean_cols) == len(clean_levels) == len(clean_min_len), (
        "Clean cols, levels, and min-lens must be give in in same quanity"
    )
    assert not (rows_per_part and n_partitions), 'Define partition split with one or the other'

    try:
        # Setup
        os.environ["POLARS_MAX_THREADS"] = f"{max_threads}"
        import polars as pl

        def measure_lf(lf: pl.LazyFrame, run=False) -> float:
            if run:
                return lf.select(pl.len()).collect(engine="streaming").item()
            else:
                return float("nan")

        if not name:
            destination = Path(origin).parent / f"{str(Path(origin).stem)}_preprocessed"
        else:
            destination = Path(origin).parent / name

        os.makedirs(destination)

        lf_whole: pl.LazyFrame = pl.scan_parquet(
            list(origin.glob("*.par*")), extra_columns="ignore"
        )


        if filt_license:
            lf_whole = lf_whole.filter(pl.col("is_license_safe"))
        else:
            logger.warning("Including non-permissive licenses")
        if start_date is not None:
            lf_whole = lf_whole.filter((pl.col("publication_date") >= start_date))
        else:
            logger.warning("No start date set")

        if end_date is not None:
            lf_whole = lf_whole.filter((pl.col("publication_date") < end_date))
        else:
            logger.warning("No end date set")
        if field_id:
            lf_whole = lf_whole.filter(pl.col("field_id").is_in(field_id))
        else:
            logger.warning("No field id filter set")

        if languages:
            lf_whole = lf_whole.filter(pl.col("language").is_in(languages))
        else:
            logger.warning("No Language filters set")

        if types:
            lf_whole = lf_whole.filter(pl.col("type").is_in(types))
        else:
            logger.warning("No document type filters set")

        n_rows: float = measure_lf(lf_whole, True)
        if n_partitions:
            rows_per_part = math.ceil(n_rows / n_partitions)
        elif rows_per_part:
            n_partitions = math.ceil(n_rows / rows_per_part)

        lf_whole = lf_whole.with_row_index("idx").with_columns(
                (pl.col("idx") // rows_per_part).clip(0, n_partitions - 1).alias("part")
            )

        progress_bar = Progress(
                    TextColumn("[bold blue] {task.description}", justify="left"),
                    BarColumn(bar_width=40),
                    TextColumn("[task.completed]{task.completed}/{task.total}"),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeElapsedColumn(),
                    TextColumn("<"),
                    TimeRemainingColumn(),
                    speed_estimate_period=60.0 * 10,  # mins
                )
        progress_bar.start()
        progress = progress_bar.add_task("Rows", total=n_rows)

        # Main Logic
        if dry_run:
            lf_whole = lf_whole.slice(0, 500)

        for i in range(n_partitions):
            lf = lf_whole.filter(pl.col("part") == i).drop(["idx", "part"])

            for col in replace_non_permissive_cols:
                logger.info(f"Replacing non-permissive {col} with {None}")
                lf = lf.with_columns(
                    pl.when(pl.col("is_license_safe") == False)
                    .then(pl.lit(None, dtype=lf.schema[col]))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

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
                    logger.info(f"Dropped {l1 - l2:,} in {col} at clean lvl {level}")
            elif i == 0:
                logger.warning("No cols selected for cleaning")

            if drop_na_cols:
                for col in drop_na_cols:
                    lf = lf.drop_nulls(subset=col)
            elif i == 0:
                logger.warning("No columns set to drop nulls")


            if embed_model:
                if i == 0:
                    logger.info(f"Embedding {', '.join(embed_cols)} with {embed_model}")
                lf = (
                    lf.drop_nulls(embed_cols)
                    .with_columns(to_embed=(
                        bos_token +
                        pl.concat_str([pl.col(col) for col in embed_cols], separator=eos_token+bos_token)
                        + eos_token)
                    ).with_columns(
                        pl.col('to_embed')
                        .map_batches(embed_batch, return_dtype=pl.Array(pl.Float32, width=model.config.hidden_size))
                        .alias(f"{' '.join(embed_cols)}_embedding")
                    ).drop('to_embed')
                )

            if tokeniser:
                if i == 0:
                    logger.info(f"Tokenising {', '.join(tokenise_cols)} with {tokeniser}")
                lf = tokenise_step(

                    lf=lf,
                    tokeniser_path=tokeniser,
                    columns=tokenise_cols,
                )

            if dry_run:
                print(lf.collect())
                break

            output_fname = destination / f"part_{i}.parquet"
            lf.sink_parquet(
                output_fname,
                statistics=True,
                compression="zstd",
                compression_level=1,
            )
            n_written = pl.scan_parquet(output_fname).select(pl.len()).collect(engine="streaming").item()
            progress_bar.update(progress, advance=n_written)


        os.makedirs(destination, exist_ok=True)
        metadata = {k: str(v) for k, v in ctx.params.items()}
        with open(destination / "metadata.json", "w") as f:
            json.dump(metadata, f)

    except Exception as e:
        try:
            logger.error(e)
        except:
            logger.error("Failed to log error")
        finally:
            raise e

    finally:
        pass
