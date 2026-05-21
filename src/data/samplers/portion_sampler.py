# src/data/samplers/portion_sampler.py
import torch

class PortionSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: torch.utils.data.Dataset, num_samples: int):
        self.dataset = dataset 
        self.num_samples = num_samples

    def __iter__(self):
        # Generate random indices for the whole dataset
        indices = torch.randperm(len(self.dataset))
        # Return only a slice of them
        return iter(indices[:self.num_samples].tolist())

    def __len__(self):
        return self.num_samples
