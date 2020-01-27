import torch
from torch import nn
import numpy as np
from torch.distributions import Normal


class HeteroscedasticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.const = (1 / 2) * np.log(2 * np.pi)

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        assert preds.shape[-1] % 2 == 0
        assert preds.shape[-1] / 2 == target.shape[-1]

        # first mean of all output dims then var of all output dims

        outut_dim = int(preds.shape[-1] / 2)
        mean = preds[:, :outut_dim]
        std = preds[:, outut_dim:]

        # bayesian nll
        dist = Normal(mean, std)
        loss = (-dist.log_prob(target)).mean()

        return loss
