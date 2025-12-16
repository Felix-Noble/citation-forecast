from typing import Callable
import torch

class ClassificationTracker:
    """
        Classification Metrics Tracker:

        Stores intermediate outputs on GPU, calcualtes results, displays them

        args:        
            output_shape: shape out output by model
            output_buffer_size: n. of outputs that will be stored in 'fast' buffer
            output_max_size: n. of outputs that will be stored before moving to CPU

    """
    def __init__(self,
                 output_shape: tuple[int, ...],
                 output_buffer_size: int,
                 n_examples: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        self.output_shape: tuple[int, ...] = output_shape
        self.device: torch.device = device
        self.buffer_cursor: torch.Tensor = torch.tensor(0, device=self.device, dtype=torch.int32)
        self.output_buffer_size: torch.Tensor = torch.tensor(output_buffer_size, device=self.device, dtype=torch.int32)
        
        # Ouptut buffers 
        self.output_buffer: torch.Tensor = torch.empty((output_buffer_size, *output_shape))
        self.output_list: list[torch.Tensor] = [] 
            
        # GPU stream (distinct from main stream)
        self.stream:torch.cuda.Stream = torch.cuda.Stream()

        self.store_output: Callable[[torch.Tensor], None]
        # select store_output method 
        if output_buffer_size >= n_examples:
            self.store_output = torch.compile(self._store_output_fast, fullgraph=True) # pyright: ignore[reportUnknownMemberType] 
        else:
            self.store_output = self._store_output_cpu 

    def _flush_buffer(self) -> None:
        " Flushed bufffer, resets cursor"
        self.output_list.append(self.output_buffer.cpu())
        self.output_buffer = torch.empty_like(self.output_buffer)
        _ = self.buffer_cursor.fill_(0)

    def _store_output_fast(self, output: torch.Tensor) -> None:
        " Stores outputs in buffer only "
        with torch.cuda.stream(self.stream):
            self.output_buffer[self.buffer_cursor[0]] = output.detach()
            _ = self.buffer_cursor.add_(1)

    def _store_output_cpu(self, output: torch.Tensor) -> None:
        " Stores outputs in buffer, syncs to CPU when full and flushes "
        with torch.cuda.stream(self.stream):
            full = self.buffer_cursor >= self.output_buffer_size

            if full:
                self._flush_buffer()
       
            self.output_buffer[self.buffer_cursor[0]] = output.detach()
            _ = self.buffer_cursor.add_(1)

     
