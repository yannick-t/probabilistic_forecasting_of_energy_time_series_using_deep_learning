def default_plt_style(plt):
    plt.style.use('seaborn-whitegrid')


def default_fig_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
