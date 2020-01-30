import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal
import numpy as np

from util.visualization.plt_styling import default_fig_style, default_plt_style


def pit_sanity_check():
    # ideal forecaster with 10000 samples
    dist_mu = Normal(0, 1)
    mu = dist_mu.sample([10000])
    real_sigma = torch.Tensor([1]).repeat(mu.shape[0])

    real_dist = Normal(mu, real_sigma)
    obs = real_dist.sample([1]).squeeze()

    probabilistic_calibration(mu, real_sigma**2, obs)


def probabilistic_calibration(pred_y_mean, pred_y_var, y_true):
    default_plt_style(plt)
    n_bins = 20

    mean = torch.Tensor(pred_y_mean).cpu()
    var = torch.Tensor(pred_y_var).cpu()
    y = torch.Tensor(y_true).cpu()

    dist = Normal(mean, var.sqrt())
    # dist = Normal(torch.Tensor([0]).repeat(10000), torch.Tensor([2]).repeat(10000))
    pt = dist.cdf(y)

    ax = plt.subplot(1, 1, 1)
    default_fig_style(ax)

    pt = pt.squeeze().numpy()

    ax.hist(pt, n_bins, color='lightblue', density=True, weights=np.zeros_like(pt) + 1. / pt.size)

    plt.show()
