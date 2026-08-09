import torch

import models

clss = models.EmbedGraphClass
model = clss.config(
    vocab_size=201_088,
    dtype=torch.float32,
    n_heads=4,
    input_seq_len=150,
    latent_seq_len=1000,
    # abstracted_seq_len=100,
    # abstraction_heads=6,
    # n_abstractions=1,
    n_layers=24,
    # n_abstractions=1,
    embed_dim=768,
    hidden_dim=768 * 4,
    n_out=1,
    causal_mask=False,
    #    n_params_out: int = 0,
    dropout=0.05,
)
# n_out = model.n_out
dtype = torch.float32
