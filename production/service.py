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
    image=image,
    max_containers=1,
    timeout=120,
    gpu="T4",
    volumes={
        '/weights': weights_volume,
        '/tokenizer': tokenizer_volume
    }
)

class InferenceWrapper:
    @modal.enter()
    def setup(self):
        # Model
        
        #sys.path.append('./models/general/model')
        from models.general.model import Model, config

        self.device=torch.device('cuda')
        self.max_len = config.data.max_len

        self.model = Model(config.model, torch.device('cuda'), torch.float32)
        self.model.eval()
        # Weights
        self.model.load_state_dict(torch.load('/weights/weights.pt', weights_only=True))

        # Tokenizer
        from transformers import AutoTokenizer
        tokenizer_name = 'openai/gpt-oss-120b'
        tokenizer_path ='/tokenizer/openai-gpt-oss-120b'
        if os.path.exists(f'/tokenizer/{tokenizer_path}'):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token_id = 0
            self.tokenizer.save_pretrained(tokenizer_path)
            tokenizer_volume.commit() 

    @modal.batched(max_batch_size=4, wait_ms=500)
    def process_request(self, prompt: list[str] | list[list[str]]):
        if isinstance(prompt[0], list):
            unwrapped = []
            for ls in prompt:
                unwrapped += ls
            prompt = unwrapped
        try: 
            token_out = self.tokenizer(
                prompt, 
                add_special_tokens=True,
                return_tensors="pt",
                max_len=self.max_len,
                padding=True,
                truncate=True,
                padding_side='right',
            )

            tokens = token_out['input_ids'].to(self.device)
            mask = token_out['attention_mask'].to(self.device)
            mask = mask.bool().unsqueeze(1).expand(-1, mask.size(-1), -1)

            with torch.inference_mode():
                logits = self.model(tokens, mask)
                probs = torch.softmax(logits, dim=-1)

            return probs.cpu().tolist()

        except Exception as e:
            return [str(e) for _ in range(len(prompt))]

@app.function(
    image=image,
    max_containers=1,
    timeout=120,
)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
async def predict(data: dict):
    # This creates an instance of the class and calls the batched method
    # Modal automatically groups these calls into 'handle_batch'
    instance = InferenceWrapper()
    probs = instance.process_request.remote([data["prompt"]])
    return {"probs": probs}
