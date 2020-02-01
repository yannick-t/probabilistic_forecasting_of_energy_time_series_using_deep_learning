import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.distributions import Normal


def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))


def crps(mu, sigma, y):
    # Gneiting, T., Raftery, A. E., Westveld III, A. H., & Goldman, T. (2005).
    # Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation.
    # Monthly Weather Review, 133(5), 1098-1118.
    # Formula 5

    sx = (y - mu) / sigma

    normal = multivariate_normal(0, 1)
    pdf = normal.pdf(sx).reshape([-1, 1])
    cdf = normal.cdf(sx).reshape([-1, 1])

    assert pdf.shape == cdf.shape == sx.shape == y.shape

    crps = sigma * (sx * (2 * cdf - 1) + 2 * pdf - (1 / np.sqrt(np.pi)))

    assert not np.any(np.isnan(crps))

    crps = np.mean(crps, axis=0)

    return crps


def mape(y_pred, y_true):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true), axis=0)


def log_likelihood(mu, sigma, y):
    mu = torch.Tensor(mu).cpu()
    sigma = torch.Tensor(sigma).cpu()
    y = torch.Tensor(y).cpu()

    dist = Normal(mu, sigma)
    ll = dist.log_prob(y)

    return ll.mean(0).numpy()
