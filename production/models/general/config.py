from dataclasses import dataclass

top_k = (300 for _ in range(8))
top_k = (*top_k, 256, 128, 64, 32, 16, 8)
n_layers = len(top_k)
selector_heads = tuple(8 for _ in range(n_layers))
process_heads = tuple(8 for _ in range(n_layers))

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = 'h_attn'
    vocab_size: int = 201_088
    pad_token: int = 0
    outer_heads: int = 2
    top_k: tuple[int, ...] = top_k
    selector_heads: tuple[int, ...] = selector_heads
    process_heads: tuple[int, ...] = process_heads
    n_layers: int = n_layers
    embed_dim: int = 256
    hidden_dim: int = 512
    n_out: int = 3
    dropout: float = 0.05

class DataConfig:
    max_len: int = 300

class Config:
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    
config = Config()
