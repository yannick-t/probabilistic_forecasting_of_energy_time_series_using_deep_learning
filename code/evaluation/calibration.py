import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal
import numpy as np


def probabilistic_calibration_multiple(names, pred_y_mean, pred_y_var, y_true, pred_y_mean_comp=None, pred_y_var_comp=None):
    count = len(pred_y_mean)
    max_columns = 3
    rows = int(count / (max_columns + 1)) + 1
    fig, axes = plt.subplots(rows, min(count, max_columns), sharey='row', figsize=(6, rows * 1.8))
    for counter, (name, pmean, pvar) in enumerate(zip(names, pred_y_mean, pred_y_var)):
        ax = axes[counter]
        ax.set_title(name)
        if pred_y_var_comp is not None:
            probabilistic_calibration(pmean, pvar, y_true, ax, pred_y_mean_comp[counter], pred_y_var_comp[counter])
        else:
            probabilistic_calibration(pmean, pvar, y_true, ax)
    fig.text(0.016, 0.5, 'Relative Frequency', va='center', rotation='vertical')
    plt.subplots_adjust(left=0.1)


def marginal_calibration_multiple(names, pred_y_mean, pred_y_var, y_true):
    count = len(pred_y_mean)
    max_columns = 3
    rows = int(count / (max_columns + 1)) + 1
    fig, axes = plt.subplots(rows, min(count, max_columns), sharey='row', figsize=(6, rows * 1.8))
    for counter, (name, pmean, pvar) in enumerate(zip(names, pred_y_mean, pred_y_var)):
        ax = axes[counter]
        ax.set_title(name)

        marginal_calibration(pmean, pvar, y_true, ax)
    fig.text(0.014, 0.5, 'CDF Difference', va='center', rotation='vertical')
    plt.subplots_adjust(left=0.11)


def probabilistic_calibration(pred_y_mean, pred_y_var, y_true, ax, pred_y_mean_comp=None, pred_y_var_comp=None):

    # evaluate probabilistic calibration with PIT histogram
    n_bins = 20

    pt = pit_calc(pred_y_mean, pred_y_var, y_true)
    ax.hist(pt, n_bins, color='lightblue', density=True, weights=np.zeros_like(pt) + 1. / pt.size, rwidth=0.9)
    if pred_y_var_comp is not None:
        pt1 = pit_calc(pred_y_mean_comp, pred_y_var_comp, y_true)
        ax.hist(pt1, n_bins, color='orange', alpha=0.2, density=True, weights=np.zeros_like(pt1) + 1. / pt1.size, rwidth=0.9)

    # 1 line
    px = np.arange(0, 1, 0.01)
    ax.plot(px, np.repeat(1, px.shape), color='lightgray', linestyle="--", alpha=0.75)

    ax.margins(0, 0.06)


def pit_calc(means, vars, targets):
    mean = torch.Tensor(means).cpu().squeeze()
    var = torch.Tensor(vars).cpu().squeeze()
    y = torch.Tensor(targets).cpu().squeeze()

    dist = Normal(mean, var.sqrt())
    pt = dist.cdf(y)

    pt = pt.squeeze().numpy()

    return pt


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
    mean = torch.Tensor(pred_y_mean).cpu().squeeze()
    var = torch.Tensor(pred_y_var).cpu().squeeze()
    y = torch.Tensor(y_true).cpu().squeeze()

    dist = Normal(mean, var.sqrt())

    # calc and display difference of empirical cdf an avg predictive cdf (like in
    # "Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2), 243-268.")

    emp_cdf = lambda x: (y <= x.unsqueeze(-1)).double().mean(-1)
    avg_pred_cdf = lambda x: dist.cdf(x.unsqueeze(-1)).mean(-1)

    min_x = y.min()
    max_x = y.max()
    eps = np.abs(max_x - min_x) * 0.05
    min_x = min_x - eps
    max_x = max_x + eps

    step = (max_x - min_x) / 1000
    plt_x = torch.arange(min_x, max_x, step)

    pcdf = avg_pred_cdf(plt_x)
    ecdf = emp_cdf(plt_x)

    dif = pcdf - ecdf

    ax.plot(plt_x, dif, color='lightblue')

    ax.plot(plt_x, np.repeat(0, plt_x.shape), color='lightgray', linestyle="--", alpha=0.75)


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
