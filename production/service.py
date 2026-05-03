import modal
app = modal.App('citescout-general')

tokenizer_volume = modal.Volume.from_name('tokenizer-gpt-oss-120b', create_if_missing=True)
weights_volume = modal.Volume.from_name('general-weights', create_if_missing=True)

inference_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install("torch<3", "transformers", "pydantic", "fastapi[standard]")
    .add_local_python_source('models')
)

server_image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('fastapi[standard]')
)

@app.cls(
    gpu="T4",
    image=inference_image,
    max_containers=1,
    timeout=30,
    scaledown_window=2,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={
        '/weights': weights_volume,
        '/tokenizer': tokenizer_volume
    }
)

class GPUInference:
    @modal.enter(snap=True)
    def setup(self):
        # Model
        import os
        import torch
        from models.general import Model, config
        
        # localise imports
        self.inference_mode = torch.inference_mode

        self.device=torch.device('cuda')
        self.max_len = config.model.max_len

        self.model = Model(config.model, torch.device('cuda'), torch.float32)
        print('Model formed')
        self.model.eval()

        # Weights
        print(os.listdir('/weights'))
        self.model.load_state_dict(torch.load('/weights/weights.pt', weights_only=True))
        print('Model Loaded')

        # Tokenizer
        from transformers import AutoTokenizer
        tokenizer_path ='/tokenizer'
        if os.path.exists('/tokenizer'):
            print('Loading tokeniser from volume')
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token_id = 0
        else:
            print('Tokenizer not found')
            print(os.listdir('/tokenizer'))
        
        # Sync & Clear for snapshot
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        print("Snapshot state ready.") 

    @modal.batched(max_batch_size=4, wait_ms=200)
    def _batch_inference(self, prompt: list[str] | list[list[str]]):
        try: 
            print('Tokenising')
            token_out = self.tokenizer(
                prompt, 
                add_special_tokens=False,
                return_tensors="pt",
                max_length=self.max_len,
                padding='max_length',
                truncate=True,
                padding_side='right',
            )

            print('Tokens made:', token_out['input_ids'].shape)
            print('Max len:', self.max_len)
            tokens = token_out['input_ids'].to(self.device)
            mask = token_out['attention_mask'].to(self.device)
            mask = mask.bool().unsqueeze(1).expand(-1, mask.size(-1), -1)
            print('Moved data to GPU')
            with self.inference_mode():
                logits, probs, sigma = self.model(tokens, mask)
            print('Inference complete')
            return probs.cpu().tolist()

        except Exception as e:
            print(str(e))
            print(f'ERROR: {e}')
            raise e
            return [str(e) for _ in range(len(prompt))]

@app.cls(
    image=server_image,
    max_containers=5,
    scaledown_window=3600,
    timeout=30,
)
@modal.concurrent(max_inputs=1000)
class Server:
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def handle(self, data: dict):
         # This creates an instance of the class and calls the batched method
         # Modal automatically groups these calls into 'handle_batch'
        instance = GPUInference()
        probOnePlus = await instance._batch_inference.remote.aio(data["promptStr"])
        return {"probOnePlus": probOnePlus}
