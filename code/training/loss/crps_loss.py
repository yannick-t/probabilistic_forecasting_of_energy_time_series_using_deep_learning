import torch
from torch import nn
import numpy as np
from torch.distributions import Normal


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

        # use sigmoid to be numerically stable and not depend on activation functions of nn
        sigma = torch.sigmoid(sigma)

        # crps
        # Gneiting, T., Raftery, A. E., Westveld III, A. H., & Goldman, T. (2005).
        # Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation.
        # Monthly Weather Review, 133(5), 1098-1118.
        # Formula 5
        sx = (target - mu) / sigma

        normal = Normal(torch.Tensor([0]).to(preds.device),
                        torch.Tensor([1]).to(preds.device))
        pdf = normal.log_prob(sx).exp()
        cdf = normal.cdf(sx)

        assert pdf.shape == cdf.shape == sx.shape == target.shape

        crps = sigma * (sx * (2 * cdf - 1) + 2 * pdf - self.const.to(preds.device))

        assert crps.shape == target.shape

        return crps.mean(0)
