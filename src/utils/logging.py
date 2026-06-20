# src/utils/logging.py
import logging
import os
from pathlib import Path

from rich.logging import RichHandler


def setup_logger(logger: logging.Logger):
    from config import logging as config

    if not os.path.exists(Path(os.getcwd()) / "logs"):
        os.makedirs(Path(os.getcwd()) / "logs")

    if getattr(logger, "_has_been_configured", False):
        return

    level_console = config.console
    level_file = config.file
    try:
        logger.setLevel(level_console)
    except ValueError as e:
        raise ValueError(f"{e} - change config value")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(Path("logs") / f"{logger.name}.log")
    file_handler.setLevel(level_file)
    file_handler.setFormatter(file_formatter)

    console_handler = RichHandler(markup=True)
    console_handler.setLevel(
        level_console
    )  # Log everything to the console for development.

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    setattr(logger, "_has_been_configured", True)

    return logger
