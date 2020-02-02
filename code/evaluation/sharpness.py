from matplotlib import pyplot as plt
from util.visualization.plotting import default_plt_style, default_fig_style
from scipy.special import erfinv
import numpy as np


def sharpness_plot_multiple(names, pred_y_var):
    ax = plt.subplot(1, 1, 1)
    ax.set_ylabel('Width')

    default_fig_style(ax)

    sharpness_plot_(pred_y_var, ax, names)

    plt.show()


def sharpness_plot(pred_y_var, ax, scaler=None):
    sharpness_plot_([pred_y_var], ax, scaler=scaler)


def sharpness_plot_(pred_y_var, ax, names=None, scaler=None):
    q_5 = np.sqrt(2) * erfinv(0.5)
    q_9 = np.sqrt(2) * erfinv(0.9)

    pred_y_var = np.array(pred_y_var)

    quantiles = np.empty([pred_y_var.shape[0] * 2, 5])

    for counter, pred_var in enumerate(pred_y_var):

        pred_std = np.sqrt(pred_var)

        widths_5 = q_5 * pred_std
        widths_9 = q_9 * pred_std

        # display 5, 25, 50, 75, 95 percentile / quantile to assess sharpness in a heteroscedastic scenario (like in
        # "Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
        # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2), 243-268.")
        quantiles_5 = np.quantile(widths_5, [0.05, 0.25, 0.5, 0.75, 0.95], axis=-2)
        quantiles_9 = np.quantile(widths_9, [0.05, 0.25, 0.5, 0.75, 0.95], axis=-2)

        if scaler is not None:
            quantiles_5 = scaler.inverse_transform(quantiles_5)
            quantiles_9 = scaler.inverse_transform(quantiles_9)

        quantiles[(counter * 2)] = quantiles_5.squeeze()
        quantiles[(counter * 2) + 1] = quantiles_9.squeeze()

    ax.boxplot([q for q in quantiles])
    if names is not None:
        ax.set_xticklabels(np.concatenate([[name + ' 50%', name + ' 90%'] for name in names]))
    else:
        ax.set_xticklabels(['50%', '90%'])


def sharpness_avg_width(pred_var):
    q_5 = np.sqrt(2) * erfinv(0.5)
    q_9 = np.sqrt(2) * erfinv(0.9)

    pred_std = np.sqrt(pred_var)

    widths_5 = q_5 * pred_std
    widths_9 = q_9 * pred_std

    return widths_5.mean(), widths_9.mean()

