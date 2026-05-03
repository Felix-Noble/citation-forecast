import os
import modal
import torch
app = modal.App('citescout-general')

tokenizer_volume = modal.Volume.from_name('tokenizer-gpt-oss-120b', create_if_missing=True)
weights_volume = modal.Volume.from_name('general-weights', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.13')
    .pip_install("torch<3", "transformers", "pydantic", "fastapi")
    .add_local_python_source('models')
)

@app.cls(
    gpu="T4",
    image=image,
    max_containers=1,
    timeout=5,
    allow_concurrent_inputs=100,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    volumes={
        '/weights': weights_volume,
        '/tokenizer': tokenizer_volume
    }
)

class GPUInference:
    @modal.enter()
    def setup(self):
        # Model
        
        #sys.path.append('./models/general/model')
        from models.general.model import Model, config

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

    @modal.batched(max_batch_size=4, wait_ms=200)
    def _batch_inference(self, prompt: list[str] | list[list[str]]):
        if isinstance(prompt[0], list):
            print('Unwrapping prompt')
            unwrapped = []
            for ls in prompt:
                unwrapped += ls
            prompt = unwrapped
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
            with torch.inference_mode():
                logits, probs, sigma = self.model(tokens, mask)
            print('Inference complete')
            return probs.cpu().tolist()

        except Exception as e:
            print(str(e))
            print(f'ERROR: {e}')
            raise e
            return [str(e) for _ in range(len(prompt))]

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def handle_request(self, data: dict):
        # This creates an instance of the class and calls the batched method
        # Modal automatically groups these calls into 'handle_batch'
        probOnePlus = self._batch_inference([data["promptStr"]])
        return {"probOnePlus": probOnePlus}

