from .model import H_ATTN as model
from .config import config
from pathlib import Path
weight_path = str(Path(__file__).parent / 'weights')
__all__ = ['model', 'config', 'weight_path']
