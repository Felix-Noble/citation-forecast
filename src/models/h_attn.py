from pydantic import BaseModel, PositiveInt
import torch
from torch import Tensor
import torch.nn as nn

class ModelConfig(BaseModel):
    outer_heads: PositiveInt
    top_k: list[int]
    selector_heads: list[int]
    process_heads: list[int]
    n_layers: PositiveInt
    vocab_size: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt
    n_out: PositiveInt

class LayerConfig(BaseModel):
    outer_heads: PositiveInt
    k: PositiveInt
    selector_heads: PositiveInt
    process_heads: PositiveInt
    embed_dim: PositiveInt
    hidden_dim: PositiveInt

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.QKV_proj = nn.Linear(dim_in, hidden_dim * n_heads * 3)
        self.head = nn.Linear(hidden_dim * n_heads, dim_out) 

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        QKV = self.QKV_proj(x)
        Q, K, V = torch.chunk(QKV, 3, dim=-1)
        Q = Q.reshape(B, T, -1, self.n_heads).permute(0, 3, 1, 2)
        K = K.reshape(B, T, -1, self.n_heads).permute(0, 3, 1, 2)
        V = V.reshape(B, T, -1, self.n_heads).permute(0, 3, 1, 2)
        
        print(Q.shape, K.shape, V.shape)

        out = nn.functional.scaled_dot_product_attention(Q, K, V, mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1))
        
        out = out.transpose(1, 2).contiguous().flatten(-2)
        out = self.head(out)
        return out

class MLP(nn.Module):
    def __init__(self, config: LayerConfig):
        super().__init__() 
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.embed_dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)

class SelectiveAttn(nn.Module):
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.relevance_selector = MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, config.hidden_dim, config.selector_heads)
        self.head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.GELU(),
        )

    def forward(self, x:Tensor, mask: Tensor) -> Tensor:
        out = self.relevance_selector(x, mask)
        out = nn.functional.softmax(out, dim=-1) 
        out = self.head(out)
        return out

class HAttnBlock(nn.Module):
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.selective_attn = nn.ModuleList(
                [ SelectiveAttn(config) for _ in range(config.outer_heads) ]
        )
        self.process_attn = nn.ModuleList(
                [ MultiHeadSelfAttn(config.embed_dim, config.hidden_dim, config.hidden_dim, config.process_heads) for _ in range(config.outer_heads) ]
        )
        self.mlp = nn.ModuleList(
                [ MLP(config) for _ in range(config.outer_heads) ]
        ) 
    def forward(self, x:Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        OB, B, T, C = x.shape
        out = torch.empty_like(x)
        mask_out = torch.empty_like(mask)
        for i in range(self.config.outer_heads):
            scores = self.selective_attn[i](x[i], mask[i])  
            _,  inds = torch.topk(scores.squeeze(-1), self.config.k, dim=-1)
            inds, _ = torch.sort(inds, dim=1)
            print('mask inside', mask[i].shape)
            print(inds.shape)
            ordered_scores = torch.gather(
                scores, 
                dim=1, 
                index=inds.unsqueeze(-1)
            )
            selected_tokens = torch.gather(
                x[i], 
                dim=1, 
                index=inds.unsqueeze(-1).expand(-1, -1, C)
            )
             
            selected_mask = torch.gather(
                mask[i],
                dim=0,
                index=inds.unsqueeze(-1).expand(-1, -1, self.config.k)
            )
            print('selected mask', selected_mask.shape)
            mask_out[i] = selected_mask.clone()
            out[i] = self.mlp[i](
                self.process_attn[i](selected_tokens * ordered_scores, selected_mask)
            )
        return out.contiguous(), mask_out

class H_ATTN(nn.Module):
    MODEL_NAME = 'h_attn'
    def __init__(self, config: ModelConfig, device, dtype):
        super().__init__()
        assert len(config.process_heads) == config.n_layers
        assert len(config.selector_heads) == config.n_layers
        assert len(config.top_k) == config.n_layers
        self.config = config 

        self.embed = nn.Embedding(config.vocab_size, config.embed_dim)
        
        layers = []
        for i in range(config.n_layers):
            layer_config = LayerConfig(
                outer_heads=config.outer_heads,
                k=config.top_k[i],
                selector_heads=config.selector_heads[i],
                process_heads=config.process_heads[i],
                embed_dim=config.embed_dim,
                hidden_dim=config.hidden_dim,
            )
            layers.append(HAttnBlock(layer_config))

        self.layers = nn.ModuleList(layers)

        self.head = nn.Linear(config.embed_dim, config.n_out)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        
        embeddings = self.embed(x)

        B, T, C = embeddings.shape

        out = embeddings.unsqueeze(0).expand(self.config.outer_heads, -1, -1, -1)
        mask = mask.unsqueeze(0).expand(self.config.outer_heads, -1, -1, -1)
        for layer in self.layers:
            print('mask shape in', mask.shape)
            delta, mask = layer(out, mask)
            out = out + delta

        out = self.head(out)
        return torch.mean(out, dim=-1)    

if __name__ == '__main__':
    config = ModelConfig(
        outer_heads = 2,
        top_k = [4,1],
        selector_heads = [3,3],
        process_heads = [3,3],
        n_layers = 2,
        vocab_size = 4 ,
        embed_dim = 3,
        hidden_dim = 10,
        n_out = 1,
    )
    
    model = H_ATTN(config, 'cpu', torch.float32)
    input = torch.ones(7, 5)
    mask = torch.ones(7, 5, 5)
    out = model(input.long(), mask)
    print(out.shape)
