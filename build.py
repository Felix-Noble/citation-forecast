#!/usr/bin/env python3
from config import config, env
from src.utils.logging import setup_logger
from logging import getLogger
from pathlib import Path
import os
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

REQUIRED_PATHS = [
    './src/types/'
]
for f in REQUIRED_PATHS:
    if not os.path.exists(f):
        logger.info(f'Initialising dir: {f}')
        os.makedirs(f)

FOUND_PATHS: tuple[str, ...] = tuple(f'"{f.stem}"' for f in env.STAGED_LOC.glob('*'))
valid_paths_type = f'''
from typing import Literal 
VALID_PATHS = Literal[{", ".join(FOUND_PATHS)}]
'''
with open('./src/types/valid_paths.py', 'w') as f:
   f.write(valid_paths_type.strip())
