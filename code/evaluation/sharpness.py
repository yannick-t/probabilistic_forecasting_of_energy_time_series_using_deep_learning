from matplotlib import pyplot as plt
import seaborn as sns
from scipy.special import erfinv
import numpy as np
import pandas as pd


def sharpness_plot_multiple(names, pred_y_var):
    fig, ax = plt.subplots(figsize=(6, 3.0))
    plt.subplots_adjust(bottom=0.16, left=0.1, top=0.93)
    ax.set_ylabel('Width')

    sharpness_plot_(pred_y_var, ax, names)
    # plt.subplots_adjust(left=0.1)


def sharpness_plot(pred_y_var, ax, scaler=None):
    sharpness_plot_([pred_y_var], ax, scaler=scaler)


def sharpness_plot_histogram(pred_y_var, ax):
    n_bins = 60
    ax.hist(pred_y_var, n_bins, histtype='step', color='lightblue',
            density=True, weights=np.zeros_like(pred_y_var) + 1. / pred_y_var.size)
    ax.margins(0, 0.06)


def sharpness_plot_(pred_y_var, ax, names=None):
    quantiles = [0.9]

    pred_y_var = np.array(pred_y_var)

    columns = ['Method', 'Quantile', 'Width']
    widths = pd.DataFrame(columns=columns)

    for counter, pred_var in enumerate(pred_y_var):

        pred_std = np.sqrt(pred_var)
        for q_counter, q in enumerate(quantiles):
            if names is not None:
                name = names[counter]
            else:
                name = ''

            quantile_widths = np.sqrt(2) * erfinv(q) * pred_std
            n_widths_df = pd.DataFrame(columns=columns, index=range(quantile_widths.shape[0]))

            n_widths_df.loc[:, 'Method'] = np.repeat(name, quantile_widths.shape[0])
            n_widths_df.loc[:, 'Quantile'] = np.repeat(q, quantile_widths.shape[0])
            n_widths_df.loc[:, 'Width'] = quantile_widths.squeeze()

            widths = pd.concat([widths, n_widths_df], axis=0)

    # display 5, 25, 50, 75, 95 percentile / quantile as boxplot to assess sharpness in a heteroscedastic scenario
    # (like in
    # "Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness.
    # Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(2), 243-268.")
    sns.boxplot(x='Quantile', y='Width', hue='Method', data=widths, whis=[5, 95], ax=ax, showfliers=False)


def sharpness_avg_width(pred_var):
    q_5 = np.sqrt(2) * erfinv(0.5)
    q_9 = np.sqrt(2) * erfinv(0.9)

    pred_std = np.sqrt(pred_var)

    widths_5 = q_5 * pred_std
    widths_9 = q_9 * pred_std

    return widths_5.mean(), widths_9.mean()

