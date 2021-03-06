import seaborn as sns
from scipy.special import erfinv

from evaluation.evaluation_plot_util import plot_multiple
import matplotlib.pyplot as plt
import numpy as np

'''
Utility methods to evaluate the distribution of predictive standard deviations on out of distribution (o.o.d.) data
compared to predictive standard deviations on test data.  
'''


def ood_sharpness_plot_histogram_joint(pred_test_var, pred_ood_var, ax):
    sns.distplot(np.sqrt(pred_ood_var).squeeze(), ax=ax, color='orange', kde=False)
    sns.distplot(np.sqrt(pred_test_var).squeeze(), ax=ax, color='lightblue', kde=False)
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
        if name == 'Simple NN':
            cutoff = 3**2
        if name == 'Concrete':
            cutoff = 4**2
        if name == 'Deep Ens.':
            cutoff = 4**2
        if cutoff is not None:
            p_ood_var = p_ood_var[p_ood_var < cutoff]
            p_test_var = p_test_var[p_test_var < cutoff]
        ood_sharpness_plot_histogram_joint(p_test_var, p_ood_var, ax)

    plot_multiple(plot_fn, count, sharey=False)
