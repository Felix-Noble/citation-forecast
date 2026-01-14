from pydantic import BaseModel, PositiveInt 
import torch
import torch.nn as nn
from torch import Tensor

def recursive_mm(x: list[Tensor], eos_one_hot):
    n = len(x)
    if n > 2:
        return recursive_mm(x[n/2:], eos_one_hot), recursive_mm(x[:n/2], eos_one_hot)

    return x[0] @ x[1]

class ConfigSchema(BaseModel):
    model_name: str
    vocab_size: PositiveInt 
    eos_token: PositiveInt
    n_layers: PositiveInt
    embed_dim: PositiveInt
    attention_dim: PositiveInt
    n_out: PositiveInt

class R_RNN_Fast(nn.Module):
    MODEL_NAME = 'r_rnn_fast'
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

        self.attention_proj = nn.Sequential(
            nn.Linear(model_config.attention_dim, model_config.embed_dim, device=device, dtype=dtype),
            nn.GELU(),
        )
        
        self.embed_proj = nn.Sequential(
            nn.Linear(model_config.embed_dim, model_config.attention_dim, device=device, dtype=dtype),
            nn.GELU(),
        )

        self.normaliser = nn.GELU()

        self.head = nn.Linear(model_config.attention_dim, model_config.n_out, device=device, dtype=dtype)

    def eos_attention_escape(self, attention: Tensor, next_attention: Tensor, eos_one_hot_t: Tensor):
        """ Attention addition for torch.cond (true_fn) call """
        next_attention += attention.squeeze(-1) * eos_one_hot_t.unsqueeze(-1)
    
    def no_eos_func(self, attention: Tensor, next_attention: Tensor, eos_one_hot_t: Tensor):
        """ Attention addition for torch.cond (false_fn) call """
        pass

    def forward(self, x: Tensor):
        eos_one_hot = torch.zeros_like(x)
        max_eos_idx = torch.max(torch.where(x == self.config.eos_token)[1])
        eos_one_hot[x == self.config.eos_token] = 1
        embeddings = self.embed(x)
        B, T, C = embeddings.shape
        attention = torch.ones((B, self.config.attention_dim, 1), device=self.device, dtype=self.dtype)
        for layer in range(self.config.n_layers):
            embed_projections = self.embed_proj(embeddings).unsqueeze(-1)
            embed_transforms = embed_projections @ embed_projections.transpose(-1, -2)
            next_attention = torch.zeros_like(attention.squeeze(-1))
            for t in range(max_eos_idx):
                attention = embed_transforms[:, t, :, :] @ attention
                attention = nn.functional.rms_norm(attention, (attention.shape[-1], ))
                next_attention += attention.squeeze(-1) * eos_one_hot[:, t].unsqueeze(-1)

            attention = next_attention.unsqueeze(-1).clone()
            attention_projection = self.attention_proj(attention.squeeze(-1)).unsqueeze(-1)
            attention_transformation = attention_projection @ attention_projection.transpose(-1, -2)
            embeddings = attention_transformation.unsqueeze(1).expand(-1, T, -1, -1) @ embeddings.unsqueeze(-1)
            embeddings = embeddings.squeeze(-1)

        out = self.head(attention.squeeze(-1)) 
        return out

if __name__ == '__main__':
    import torch
    x = torch.rand(2, 5, 1)
    B, T, *N  = x.shape

    eos_one_hot = torch.zeros(B, T)
    eos_one_hot[0, -2] = 1
    eos_one_hot[1, -3] = 1

    eos_idx = torch.where(eos_one_hot == 1)
    sort_index = sorted(list(range(B)), key = lambda i : eos_idx[1][i]) 
    eos_idx_sorted = eos_idx[0][sort_index]

    x = x[eos_idx_sorted, :,  :]

    x = x.permute(1, 0, 2)
    eos_batch_idx = eos_idx[1][sort_index]
    t_points = []
    # only iterate until the max time point until all eos reached 
    for i, ts in enumerate(x):
        in_seq = ts
        t_points.append(ts[eos_batch_idx >= i, :])
    
    for i in range(0, T - (T%2), 2):
        print(t_points[i])
        print(t_points[i+1])

        a = t_points[i]
        b = t_points[i+1]

        print(a.shape == b.shape)
        
        # base case: launch bmm kernel for i and i+1
        # edge case: return tensor 
        # func def: snake along the function (though recursive for data dependency) via fixed indexs

