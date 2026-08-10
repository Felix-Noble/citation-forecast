from torch import Tensor, tensor, log, mean
from torch.distributions.categorical import Categorical

def norm_entropy_loss(distribution: Tensor ) -> Tensor:
    """ # Normalised Entropy loss
        ## Args:
            distribution: 1D probability distribution, shape (B, ..., N), B = batch, N = n classes
    """
    entropy = Categorical(distribution).entropy()
    max_entropy = log(tensor(distribution.shape[-1], device=distribution.device))
    loss = mean(entropy / max_entropy)
    return loss
