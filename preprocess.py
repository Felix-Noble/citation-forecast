#!/usr/bin/env python3
from config.config import config
from src.utils.logging import setup_logger
from src.preprocess import tokenise, stage_data
from logging import getLogger
from pathlib import Path
from typing import Literal
import typer

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

app = typer.Typer()
app.add_typer(tokenise.app, name='tokenise')

VALID_FILENAMES = Literal['tokenise', 'stage_data']
@app.command()
def main(filename: VALID_FILENAMES = typer.Argument(
    help='name of file to run',
)
         ) -> None:
    pass 

if __name__ == '__main__':
    app()

