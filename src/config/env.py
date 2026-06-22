# config/env.py
from pathlib import Path

TRACKING_URI: str = "http://127.0.0.1:5000"
EXPERIMENT: str = "General-2"
RAW_LOC: Path = Path("/home/fnoble/data/OpenAlex-parquet/")
STAGED_LOC: Path = Path("/home/fnoble/data/")
ARTIFACT_LOC: str = "/home/fnoble/experiment-tracking/artifacts"
