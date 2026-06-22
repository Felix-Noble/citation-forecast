import torch

import models

clss = models.TransformerClass
model = clss.config(
    vocab_size=201_088,
    dtype=torch.float32,
    n_heads=1,
    # abstracted_seq_len=100,
    # abstraction_heads=6,
    # n_abstractions=1,
    n_layers=12,
    # n_abstractions=1,
    embed_dim=8,
    hidden_dim=16,
    n_out=1,
    #    n_params_out: int = 0,
    dropout=0.05,
)
n_out = model.n_out
dtype = torch.float32
