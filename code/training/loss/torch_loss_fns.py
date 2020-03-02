import torch
from torch.distributions import Normal
import numpy as np

crps_const = 1 / torch.Tensor([np.pi]).sqrt()


def crps_torch(mean, std, target):
    # crps
    # Gneiting, T., Raftery, A. E., Westveld III, A. H., & Goldman, T. (2005).
    # Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation.
    # Monthly Weather Review, 133(5), 1098-1118.
    # Formula 5
    sx = (target - mean) / std

    normal = Normal(torch.Tensor([0]).to(sx.device),
                    torch.Tensor([1]).to(sx.device))
    pdf = normal.log_prob(sx).exp()
    cdf = normal.cdf(sx)

    assert pdf.shape == cdf.shape == sx.shape == target.shape

    crps = std * (sx * (2 * cdf - 1) + 2 * pdf - crps_const.to(sx.device))

    assert crps.shape == target.shape

    return crps.mean(0)


def bnll(mean, std, target):
    # bayesian nll
    dist = Normal(mean, std)
    return (-dist.log_prob(target)).mean(0)