from matplotlib import pyplot as plt
from scipy.special import erfinv
import numpy as np


def sharpness_plot_multiple(names, pred_y_var):
    fig, ax = plt.subplots(figsize=(10, 3.25))
    ax.set_ylabel('Width')

    sharpness_plot_(pred_y_var, ax, names)
    plt.subplots_adjust(left=0.1)

    plt.show()


def sharpness_plot(pred_y_var, ax, scaler=None):
    sharpness_plot_([pred_y_var], ax, scaler=scaler)


def sharpness_plot_histogram(pred_y_var, ax):
    n_bins = 60
    ax.hist(pred_y_var, n_bins, histtype='step', color='lightblue',
            density=True, weights=np.zeros_like(pred_y_var) + 1. / pred_y_var.size)
    ax.margins(0, 0.06)


def sharpness_plot_histogram_joint(pred_test_var, pred_ood_var, ax):
    n_bins = 25
    ax.hist(np.sqrt(pred_test_var), n_bins, histtype='stepfilled', color='lightblue', )
    ax.hist(np.sqrt(pred_ood_var), n_bins, histtype='step', color='orange')
    ax.set_yticklabels([])
    ax.margins(0, 0.06)


def sharpness_plot_histogram_joint_multiple(names, pred_test_vars, pred_ood_vars):
    count = len(names)
    fig, axes = plt.subplots(1, count, sharey='row', figsize=(9, 3.25))

    for counter, (name, p_test_var, p_ood_var) in enumerate(zip(names, pred_test_vars, pred_ood_vars)):
        ax = axes[counter]
        ax.set_title(name)

        sharpness_plot_histogram_joint(p_test_var, p_ood_var, ax)

    plt.show()


def sharpness_plot_(pred_y_var, ax, names=None, scaler=None):
    q_5 = np.sqrt(2) * erfinv(0.5)
    q_9 = np.sqrt(2) * erfinv(0.9)

    pred_y_var = np.array(pred_y_var)

    widths_5_list = np.empty([pred_y_var.shape[0], pred_y_var.shape[1]])
    widths_9_list = np.empty([pred_y_var.shape[0], pred_y_var.shape[1]])

    for counter, pred_var in enumerate(pred_y_var):

        pred_std = np.sqrt(pred_var)

        widths_5 = q_5 * pred_std
        widths_9 = q_9 * pred_std

        widths_5_list[counter] = widths_5.squeeze()
        widths_9_list[counter] = widths_9.squeeze()

    widths = np.concatenate([widths_5_list, widths_9_list])

    # display 5, 25, 50, 75, 95 percentile / quantile as boxplot to assess sharpness in a heteroscedastic scenario
    # (like in
    # "Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2), 243-268.")
    ax.boxplot([q for q in widths], showfliers=False, whis=[5, 95])
    if names is not None:
        ax.set_xticklabels(np.concatenate([[name + ' 50%' for name in names], [name + ' 90%' for name in names]]))
    else:
        ax.set_xticklabels(['50%', '90%'])


def sharpness_avg_width(pred_var):
    q_5 = np.sqrt(2) * erfinv(0.5)
    q_9 = np.sqrt(2) * erfinv(0.9)

    pred_std = np.sqrt(pred_var)

    widths_5 = q_5 * pred_std
    widths_9 = q_9 * pred_std

    return widths_5.mean(), widths_9.mean()

