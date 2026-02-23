from ._registry import loss_registry
from .wasserstein import wasserstein_loss
from .cross_entropy import CrossEntropyLoss
from config import Config, config
import torch
from torch.distributions.normal import Normal
from torch import Tensor, tensor, cuda

@loss_registry('WassersteinCrossEntropyLoss')
class WassersteinCrossEntropyLoss:
    def __init__(
            self,
            config: Config = config,
            device: torch.device = torch.device('cuda') if cuda.is_available() else torch.device('cpu'),
            ):
        self.beta = tensor(config.train.loss.beta, device=device)
        self.gamma = tensor(config.train.loss.gamma, device=device)
        self.ce_loss_fn = CrossEntropyLoss() 

    @staticmethod
    def smooth_one_hot(one_hot: Tensor, sigma: Tensor) -> Tensor:
        """ Smooths one hot vector, assumes gaussian noise """
        sigma = sigma.unsqueeze(-1)
        target_indicies = torch.argmax(one_hot, dim=-1).unsqueeze(-1)
        indicies = torch.arange(one_hot.shape[-1], device=one_hot.device).float().unsqueeze(0)
        dist = Normal(loc=target_indicies, scale=sigma)
        probs = dist.log_prob(indicies).exp()
        out =  probs / probs.sum(dim=-1, keepdim=True)
        return out

    def __call__(
        self,
        logits: Tensor,
        probs: Tensor,
        sigma: Tensor,
        target: Tensor,
        config: Config = config,
        **kwargs,
            ) ->Tensor:
        """
        # Wassterstein + Cross Entropy loss 
        ## Args: 
            distribution: 1D prob distribution (model output softmaxed)
            sigma: model confidence score (interpreted as gaussian sigma)
            target: one hot vector for target class
            beta: proportional importance of wasserstein > entropy (expected at config.train.loss.beta)
        ## Out:
            loss: cross entropy + wasserstein dist to weighted smooth distribution 
       """ 

        target_smoothed = self.smooth_one_hot(target, sigma * self.gamma)
        w_loss = wasserstein_loss(probs, target_smoothed)
        target_int = torch.argmax(target, dim=-1).long()
        c_e_loss = self.ce_loss_fn(logits, target_int)
        loss = ((1 - self.beta) * e_loss) + (self.beta * w_loss)
        return loss
