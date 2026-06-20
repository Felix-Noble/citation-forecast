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
    abstracted_seq_len=100,
    max_len=100,
    max_len_eval=100,
    embed_dim=8,
    hidden_dim=8,
    n_out=1,
    #    n_params_out: int = 0,
    dropout=0.05,
)
n_out = model.n_out
dtype = torch.float32
