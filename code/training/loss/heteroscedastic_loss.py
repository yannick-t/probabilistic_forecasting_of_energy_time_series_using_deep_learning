import torch
from torch import nn
from torch.distributions import Normal


class HeteroscedasticLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        assert preds.shape[-1] % 2 == 0
        assert preds.shape[-1] / 2 == target.shape[-1]

        # first mean of all output dims then var of all output dims

        outut_dim = int(preds.shape[-1] / 2)
        mean = preds[:, :outut_dim]
        std = preds[:, outut_dim:]

        # use softplus to be numerically stable and not depend on activation functions of nn
        softplus = torch.nn.Softplus()
        std = softplus(std)

        # bayesian nll
        dist = Normal(mean, std)
        loss = (-dist.log_prob(target)).mean(0)

        return loss
