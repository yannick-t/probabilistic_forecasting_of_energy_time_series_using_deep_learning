from matplotlib import pyplot as plt
from util.visualization.plt_styling import default_plt_style, default_fig_style
from scipy.special import erfinv
import numpy as np


def sharpness(pred_y_mean, pred_y_var, ax):
    ax.set_title('Sharpness: Predictive Interval Width')

    pred_y_std = np.sqrt(pred_y_var)

    q_5 = np.sqrt(2) * erfinv(0.5)
    q_9 = np.sqrt(2) * erfinv(0.9)

    widths_5 = q_5 * pred_y_std
    widths_9 = q_9 * pred_y_std

    # display 5, 25, 50, 75, 95 percentile / quantile to assess sharpness in a heteroscedastic scenario (like in
    # "Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2), 243-268.")
    quantiles_5 = np.quantile(widths_5.squeeze(), [0.05, 0.25, 0.5, 0.75, 0.95])
    quantiles_9 = np.quantile(widths_9.squeeze(), [0.05, 0.25, 0.5, 0.75, 0.95])

    ax.boxplot([quantiles_5, quantiles_9])
    ax.set_xticklabels(['50%', '90%'])
    ax.set_ylabel('Width')

    plt.show()

    return widths_5.mean(), widths_9.mean()
