import torch
from torch import nn
import numpy as np
from torch.distributions import Normal

from training.loss.torch_loss_fns import crps_torch


class CRPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = (1 / torch.Tensor([np.pi]).sqrt())

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        assert preds.shape[-1] % 2 == 0
        assert preds.shape[-1] / 2 == target.shape[-1]

        # first mean of all output dims then var of all output dims

        outut_dim = int(preds.shape[-1] / 2)
        mu = preds[:, :outut_dim]
        sigma = preds[:, outut_dim:]

        # use softplus to be numerically stable and not depend on activation functions of nn
        softplus = torch.nn.Softplus()
        sigma = softplus(sigma)

        return crps_torch(mu, sigma, target)
