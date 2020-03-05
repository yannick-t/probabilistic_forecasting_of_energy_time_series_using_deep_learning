import seaborn as sns
from scipy.special import erfinv

from evaluation.evaluation_plot_util import plot_multiple
import matplotlib.pyplot as plt
import numpy as np


def ood_timeframe_multiple(names, start_time, end_time, pred_means, pred_vars, timestamp, y_true):
    count = len(pred_means)

    def plot_fn(counter, ax):
        name = names[counter]
        pmean = pred_means[counter]
        pvar = pred_vars[counter]

        ax.set_title(name)

        ood_timeframe(start_time, end_time, pmean, pvar, timestamp, y_true, ax)

    fig = plot_multiple(plot_fn, count)

    fig.text(0.016, 0.5, 'Load', va='center', rotation='vertical')
    plt.subplots_adjust(left=0.1)


def ood_timeframe(start_time, end_time, pred_mean, pred_var, timestamp, y_true, ax):
    x = timestamp

    ax.plot(x, y_true.squeeze(), color='lightblue', alpha=0.4)
    ax.plot(x, pred_mean.squeeze(), color='orange')

    ax.set_xlim(start_time, end_time)

    # 90% conf intervals
    conf = np.sqrt(np.sqrt(2) * erfinv(0.9) * np.sqrt(pred_var.squeeze()))
    ax.fill_between(x, pred_mean.squeeze() - conf,
                    pred_mean.squeeze() + conf,
                    alpha=0.5, color='orange')


def ood_sharpness_plot_histogram_joint(pred_test_var, pred_ood_var, ax, bw='scott'):
    sns.distplot(np.sqrt(pred_ood_var).squeeze(), ax=ax, color='orange',
                 kde=False, kde_kws={'cut': 0, 'bw': bw}
                 )
    sns.distplot(np.sqrt(pred_test_var).squeeze(), ax=ax, color='lightblue',
                 kde=False, kde_kws={'cut': 0, 'bw': bw}
                 )
    ax.set_yticklabels([])
    ax.margins(0, 0.06)


def ood_sharpness_plot_histogram_joint_multiple(names, pred_test_vars, pred_ood_vars):
    count = len(names)

    def plot_fn(counter, ax):
        name = names[counter]
        p_test_var = pred_test_vars[counter]
        p_ood_var = pred_ood_vars[counter]

        ax.set_title(name)
        # cut down data a little for clarity for some methods
        cutoff = None
        if name == 'Simple NN' or name == 'Concrete':
            cutoff = 1.5
        if cutoff is not None:
            p_ood_var = p_ood_var[p_ood_var < cutoff]
            p_test_var = p_test_var[p_test_var < cutoff]
        ood_sharpness_plot_histogram_joint(p_test_var, p_ood_var, ax)

    plot_multiple(plot_fn, count, sharey=False)
