from pydantic import BaseModel, PositiveInt 
import torch
import torch.nn as nn
from torch import Tensor

def activation_func(x):
    return nn.functional.rms_norm(
        x,
        (x.size(-2), x.size(-1),)
        )

def recursive_mm(x, where_padding):
    n = x.shape[0]
    # case 1: single tensor
    if n < 2:
        if torch.any(where_padding):
            return ((x * 0.0) + torch.eye(x.shape[2], device=x.device)).squeeze(0) # replace with diagonal
        return x.squeeze(0)

    # case 2: > 2 tensors    
    if n > 2:
        return activation_func(
            torch.bmm(
                recursive_mm(x[:n // 2], where_padding[:n // 2]),
                recursive_mm(x[n // 2:], where_padding[n // 2:]),
            )
        )
    # base case: n == 2
    else:
        x0 = x[0] 
        x1 = x[1]
        
        where_both = where_padding[0].unsqueeze(-1).unsqueeze(-1).expand(-1, *x0.shape[1:])
        where_1 = where_padding[1].unsqueeze(-1).unsqueeze(-1).expand(-1, *x1.shape[1:])

        return torch.where(
                    where_both, ((x0 + x1) * 0.0) + torch.eye(x.shape[2], device=x.device), # replace with eye 
                        torch.where(
                            where_1, x0, activation_func(torch.bmm(x0, x1))
                        )
                )

class Projection(nn.Module):
    def __init__(self, in_dim, out_dim, device, dtype):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, out_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
    def forward(self, x):
        return self.projection(x) 

class SelectiveAttention(nn.Module):
    def __init__(self, in_dim, out_dim, device, dtype):
        super().__init__()
        self.Q_projections = nn.Sequential(
            nn.Linear(in_dim, in_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
        self.head = nn.Linear(in_dim, out_dim, device=device, dtype=dtype)

    def forward(self, embeddings, transform):
        Q = self.Q_projections(embeddings)
        out = Q @ transform.transpose(-1, -2)
        return self.head(out)

class ConfigSchema(BaseModel):
    model_name: str
    vocab_size: PositiveInt 
    eos_token: PositiveInt
    n_layers: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt

class R_RNN_Fast(nn.Module):
    MODEL_NAME = 'mmlp2'
    config_schema = ConfigSchema
    def __init__(self, 
                 model_config,
                 device: torch.device,
                 dtype: torch.dtype
                 ):
        super().__init__()
        self.config = model_config
        self.device = device
        self.dtype = dtype 
        
        self.embed = nn.Embedding(model_config.vocab_size, model_config.embed_dim, device=device, dtype=dtype)

        self.Q_projections = nn.ModuleList(
            [Projection(model_config.embed_dim, model_config.hidden_dim , device, dtype) for _ in range(model_config.n_layers)]
        ) 

        self.K_projections = nn.ModuleList(
            [Projection(model_config.embed_dim, model_config.hidden_dim , device, dtype) for _ in range(model_config.n_layers)]
        ) 

        self.mlps1 = nn.ModuleList(
            [Projection(model_config.hidden_dim, model_config.embed_dim , device, dtype) for _ in range(model_config.n_layers)]
        )

        self.attention = SelectiveAttention(model_config.embed_dim, 1, device, dtype)

        self.mlps2 = nn.ModuleList(
            [Projection(model_config.embed_dim, model_config.embed_dim , device, dtype) for _ in range(model_config.n_layers)]
        ) 
        self.head = nn.Linear(model_config.embed_dim, model_config.n_out, device=device, dtype=dtype)

    def forward(self, x: Tensor):
        indicies = torch.arange(x.shape[1], device=self.device)
        padding_begin = torch.where(x == self.config.eos_token)[1] + 1
        where_padding = torch.zeros_like(x, dtype=torch.bool)
        for i in range(x.shape[0]):
            where_padding[i, :] = indicies > padding_begin[i]

        embeddings = self.embed(x)
        B, T, C = embeddings.shape

        for i in range(self.config.n_layers):
            Q = self.Q_projections[i](embeddings).unsqueeze(-1)
            K = self.K_projections[i](embeddings).unsqueeze(-1)

            embed_matrices = (Q @ K.transpose(-1, -2)).permute(1, 0, 2, 3)

            embed_transform = recursive_mm(embed_matrices, where_padding.permute(1, 0))
            embed_transform = self.mlps1[i](embed_transform)
            embed_transform = self.mlps1[i](embed_transform.transpose(-1, -2))

            relevance = self.attention(embeddings, embed_transform)
            delta =  embeddings.unsqueeze(-2) @ embed_transform.unsqueeze(1).expand(-1, T, -1, -1)

            embeddings = embeddings + (delta.squeeze(-2) * relevance)
            embeddings = self.mlps2[i](embeddings)
            embeddings = nn.functional.rms_norm(embeddings, (embeddings.size(-1), ))  

        out = torch.mean(self.head(embeddings), dim=1)
        return out
