# config/env.py
from dataclasses import dataclass
from pathlib import Path
 
@dataclass(frozen=True)
class Env:
    TRACKING_URI: str = "http://127.0.0.1:5000"
    EXPERIMENT: str = 'NG-Generative'
    RAW_LOC: Path = Path('/home/fnoble/data/OpenAlex-parquet/')
    STAGED_LOC: Path = Path('/home/fnoble/data/')
    ARTIFACT_LOC: str = '/home/fnoble/experiment-tracking/artifacts'
