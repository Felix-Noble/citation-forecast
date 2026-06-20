import torch

import models

clss = models.TransformerClass
model = clss.config(
    vocab_size=201_088,
    pad_token_id=19999,
    dtype=torch.float32,
    n_heads=8,
    n_layers=8,
    # n_abstractions=1,
    max_len=100,  # TODO: move to data config
    embed_dim=8,
    hidden_dim=8,
    n_out=1,
    #    n_params_out: int = 0,
    dropout=0.05,
)
n_out = model.n_out
dtype = torch.float32
