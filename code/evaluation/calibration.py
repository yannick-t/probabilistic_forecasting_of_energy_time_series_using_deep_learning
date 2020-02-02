import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal
import numpy as np

from util.visualization.plotting import default_fig_style, default_plt_style


def probabilistic_calibration_multiple(names, pred_y_mean, pred_y_var, y_true):
    count = len(pred_y_mean)
    fig, axes = plt.subplots(1, count, sharey='row')
    for counter, (name, pmean, pvar) in enumerate(zip(names, pred_y_mean, pred_y_var)):
        ax = axes[counter]
        ax.set_title(name)
        default_fig_style(ax)
        probabilistic_calibration(pmean, pvar, y_true, ax)
    fig.text(0.04, 0.5, 'Relative Frequency', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


def marginal_calibration_multiple(names, pred_y_mean, pred_y_var, y_true):
    count = len(pred_y_mean)
    fig, axes = plt.subplots(1, count, sharey='row')
    for counter, (name, pmean, pvar) in enumerate(zip(names, pred_y_mean, pred_y_var)):
        ax = axes[counter]
        ax.set_title(name)
        default_fig_style(ax)

        marginal_calibration(pmean, pvar, y_true, ax)
    # fig.text(0.04, 0.5, 'Relative Frequency', va='center', rotation='vertical')
    plt.tight_layout()
    plt.show()


def probabilistic_calibration(pred_y_mean, pred_y_var, y_true, ax):

    # evaluate probabilistic calibration with PIT histogram
    n_bins = 20

    mean = torch.Tensor(pred_y_mean).cpu().squeeze()
    var = torch.Tensor(pred_y_var).cpu().squeeze()
    y = torch.Tensor(y_true).cpu().squeeze()

    dist = Normal(mean, var.sqrt())
    pt = dist.cdf(y)

    pt = pt.squeeze().numpy()

    ax.hist(pt, n_bins, color='lightblue', density=True, weights=np.zeros_like(pt) + 1. / pt.size)
    ax.margins(0, 0.06)


def interval_coverage(pred_y_mean, pred_y_var, y_true, interval):
    # "the proportion of the time that the interval contains the true value of interest"
    mean = torch.Tensor(pred_y_mean).cpu()
    var = torch.Tensor(pred_y_var).cpu()
    y = torch.Tensor(y_true).cpu()

    dist = Normal(mean, var.sqrt())
    cov = dist.cdf(y) <= interval
    cov = cov.sum() / float(cov.shape[0])

    return cov.numpy()


def marginal_calibration(pred_y_mean, pred_y_var, y_true, ax):
    default_plt_style(plt)

    mean = torch.Tensor(pred_y_mean).cpu().squeeze()
    var = torch.Tensor(pred_y_var).cpu().squeeze()
    y = torch.Tensor(y_true).cpu().squeeze()

    dist = Normal(mean, var.sqrt())

    # calc and display difference of empirical cdf an avg predictive cdf (like in
    # "Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2), 243-268.")

    emp_cdf = lambda x: (y <= x.unsqueeze(-1)).double().mean(-1)
    avg_pred_cdf = lambda x: dist.cdf(x.unsqueeze(-1)).mean(-1)

    plt_x = torch.arange(-2, 2, 0.01)

    dif = avg_pred_cdf(plt_x) - emp_cdf(plt_x)

    ax.plot(plt_x, dif, color='lightblue')


def pit_sanity_check():
    # ideal forecaster with 10000 samples
    dist_mu = Normal(0, 1)
    mu = dist_mu.sample([10000])
    real_sigma = torch.Tensor([1]).repeat(mu.shape[0])

    real_dist = Normal(mu, real_sigma)
    obs = real_dist.sample([1]).squeeze()

    ax = plt.subplot(1, 1, 1)
    probabilistic_calibration(mu, real_sigma ** 2, obs, ax)
    plt.show()
