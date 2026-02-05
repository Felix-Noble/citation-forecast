from models.general import model, weight_path, config
import os
import modal
import torch
import json

app = modal.App('citescout-general')

tokenizer_volume = modal.Volume.from_name('tokenizer-gpt-oss-120b', create_if_missing=True)
weights_volume = modal.Volume.from_name('general-weights', create_if_missing=True)

image = modal.Image.debian_slim().pip_install("torch", "transformers")

@app.cls(
    image=image,
    gpu="T4",
    volumes={
        '/weights': weights_volume,
        '/tokenizer': tokenizer_volume
    }
)
class InferenceWrapper:
    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer
        tokenizer_name = 'openai/gpt-oss-120b'
        tokenizer_path ='/tokenizer/openai-gpt-oss-120b'
        if os.path.exists(f'/tokenizer/{tokenizer_path}'):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.save_pretrained(tokenizer_path)
            tokenizer_volume.commit() 
        
        self.max_len = config.data.max_len
        self.model = model(config.model, torch.device('cuda'), torch.float32)
        self.model.load_state_dict(torch.load('/weights/weights.pt', map_location='cuda', weights_only=True))
        self.model.eval()

    @modal.method()
    def process_request(self, request: dict):
        try: 
            prompt = request.get('prompt')
            prompt = self.tokenizer.bos_token + prompt + self.tokenizer.eos_token

            tokens = self.tokenizer(prompt)
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            if len(tokens) < self.max_len:
                tokens = torch.nn.functional.pad(
                tokens, (0, self.max_len - len(tokens)), value=0
                )
            mask = (tokens != 0).bool().unsqueeze(0).expand(self.max_len, -1)

            with torch.inference_mode():
                logits = self.model(tokens, mask)
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().tolist()

        except Exception as e:
            return {'status': 'Error', 'message': str(e)}
