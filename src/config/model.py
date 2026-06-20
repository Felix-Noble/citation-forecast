import torch

import models

clss = models.Abstractor
model = clss.config(
    vocab_size=201_088,
    pad_token_id=19999,
    dtype=torch.float32,
    process_heads=1,
    n_layers=8,
    n_forward=1,
    n_abstractions=1,
    abstraction_heads=2,
    abstracted_seq_len=300,
    max_len=100,
    max_len_eval=100,
    embed_dim=4,
    hidden_dim=4,
    n_out=201_088,
    #    n_params_out: int = 0,
    dropout=0.05,
)
vocab_size: int = 201_088
pad_token_id: int = 19999
dtype: torch.dtype = torch.float32
process_heads: int = 1
n_layers: int = 8
n_forward: int = 1
n_abstractions: int = 1
abstraction_heads: int = 2
abstracted_seq_len: int = 300
max_len: int = 100
max_len_eval: int = 100
embed_dim: int = 4
hidden_dim: int = 4
n_out: int = 201_088
#    n_params_out: int = 0
dropout: float = 0.05
