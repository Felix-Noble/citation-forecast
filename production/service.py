import sys
import os
import modal
import torch

app = modal.App('citescout-general')

tokenizer_volume = modal.Volume.from_name('tokenizer-gpt-oss-120b', create_if_missing=True)
weights_volume = modal.Volume.from_name('general-weights', create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version='3.13')
    .uv_pip_install("torch<3", "transformers", "pydantic")
    .add_local_dir('./models/general/model/', remote_path='/code/model')
)

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
        # Model
        sys.path.append('/code')
        #sys.path.append('./models/general/model')
        from model import Model, config

        self.model = Model(config.model, torch.device('cuda'), torch.float32)

        self.max_len = config.data.max_len
        self.model.eval()
        # Weights
        self.model.load_state_dict(torch.load('/weights/weights.pt', map_location='cuda', weights_only=True))

        # Tokenizer
        from transformers import AutoTokenizer
        tokenizer_name = 'openai/gpt-oss-120b'
        tokenizer_path ='/tokenizer/openai-gpt-oss-120b'
        if os.path.exists(f'/tokenizer/{tokenizer_path}'):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = 0
            self.tokenizer.save_pretrained(tokenizer_path)
            tokenizer_volume.commit() 

    @modal.method()
    def process_request(self, prompt: str):
        try: 
            prompt = self.tokenizer.bos_token + prompt + self.tokenizer.eos_token

            tokens, mask = self.tokenizer(
                prompt, 
                return_tensors="pt",
                max_len=self.max_len,
                padding=True,
                padding_side='right',

            )

            mask = mask.bool().unsqueeze(0).expand(self.max_len, -1)

            with torch.inference_mode():
                logits = self.model(tokens, mask)
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().tolist()

        except Exception as e:
            return {'status': 'Error', 'message': str(e)}
