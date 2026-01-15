from pydantic import BaseModel, PositiveInt 
import torch
import torch.nn as nn
from torch import Tensor
    # test 1 : bmm the whole length (mult by eos_one_got for next atention)
    # text 2: recursively trace down each batch item individually
    # test1 compiles? ; test2 will not (variable length)
def recursive_mm_1(x, eos_one_hot, attention_mod):
    # case 1: more than two tensore 
    n = len(x)
    if n > 2:
        half_n = int(n/2)
        return torch.bmm(
            recursive_mm_1(x[:half_n], eos_one_hot, attention_mod),
            recursive_mm_1(x[half_n:], eos_one_hot, attention_mod)
        )
    # case 2: one tensor 
    elif n < 2:
        return x[0]

    # base case: bmm of two tensors
    return torch.bmm(x[0], x[1])

def parallel_matrix_product_iterative(matrices):
    curr = matrices
    while curr.shape[0] > 1:
        L = curr.shape[0]
        if L % 2 == 1:
            # Handle odd remainder by multiplying it into the first pair
            # or saving it for the end.
            remainder = curr[-1:]
            curr = curr[:-1]
        else:
            remainder = None
            
        # Batch multiply pairs
        curr = curr.view(L // 2, 2, *curr.shape[1:])
        curr = torch.matmul(curr[:, 0], curr[:, 1])
        
        if remainder is not None:
            # This is a bit simplified; logic for odd L in scans 
            # can get tricky to keep perfectly balanced.
            curr = torch.cat([curr, remainder], dim=0)
    return curr 

def recursive_mm_b(x):
    n = x.shape[0]
    if n >= 2:
        return torch.bmm(
            recursive_mm_b(x[:n // 2]),
            recursive_mm_b(x[n // 2:]),
                )
    if n < 2:
        return x.squeeze(0)

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
        eos_t_index = torch.where(x == self.config.eos_token)[1]
        eos_one_hot[x == self.config.eos_token] = 1
        embeddings = self.embed(x)
        B, T, C = embeddings.shape
        attention = torch.ones((B, self.config.attention_dim, 1), device=self.device, dtype=self.dtype)

        for layer in range(self.config.n_layers):
            embed_projections = self.embed_proj(embeddings).unsqueeze(-1)
            embed_transforms = embed_projections @ embed_projections.transpose(-1, -2)
            embed_transforms = embed_transforms.permute(1, 0, 2, 3)
            
            attention_mod = recursive_mm_b(embed_transforms)
            #attention_mod = parallel_matrix_product_iterative(embed_transforms).squeeze(0)
            attention = attention_mod @ attention

            attention_projection = self.attention_proj(attention.squeeze(-1)).unsqueeze(-1)
            attention_transformation = attention_projection @ attention_projection.transpose(-1, -2)
            embeddings = attention_transformation.unsqueeze(1).expand(-1, T, -1, -1) @ embeddings.unsqueeze(-1)
            embeddings = embeddings.squeeze(-1)

        out = self.head(attention.squeeze(-1)) 
        return out

class ModelConfig:
    model_name:str = 'R_RNN'
    vocab_size:int = 201_088
    eos_token: int = 200_002
    embed_dim: int = 512 
    attention_dim: int = 512
    n_layers: int = 2
    n_out: int = 5

if __name__ == '__main__':
    import torch
    import torch.utils.benchmark as benchmark
    from real_rnn import R_RNN 
    from torch.profiler import profile, record_function, ProfilerActivity
    config = ModelConfig()
    torch.manual_seed(2026)
    def benchmark_model(model, input_data):
        t = benchmark.Timer(
            stmt='model(input_data)',
            setup='import torch',
            globals={'model': model, 'input_data': input_data},
            num_threads=16
        )
        
        return t.blocked_autorange(min_run_time=5)

    B, T = 16, 500
    print('batch/time', B, T)
    x = torch.randint(0, 200_000, (B, T), device='cuda', dtype=torch.long)
    eos_idxs = torch.randint(200, T-1, (B,))
    for i, idx in enumerate(eos_idxs):
        x[i, -1] = config.eos_token
        #x[i, idx+1:] = 0
    torch.set_float32_matmul_precision('highest')

    test = 'new'
    run_profile = False 
    if test == 'old':
        print('Testing old\n')
        model = R_RNN(config, torch.device('cuda'), torch.float32)
    elif test == 'new':
        print('Testing new\n')
        model = R_RNN_Fast(config, torch.device('cuda'), torch.float32) 
        model = torch.compile(model, fullgraph=True, mode='default')

    if run_profile:
        supported = torch.profiler.supported_activities()
        print(f"Supported Activities: {supported}")
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True, # Optional: if you want to see allocations
            with_stack=True      # Optional: correlates kernels to Python code lines
        ) as prof:
            with record_function("test"):
                # Perform your operations here
                out = model(x)
                torch.cuda.synchronize() # Optional: keeps everything in the trace window
        prof.export_chrome_trace("trace.json")

    else:
        with torch.no_grad():
            model.eval()
            results = benchmark_model(model, x)
            print(type(model).__name__)
            print(results)
        
