import matplotlib.pyplot as plt
import math


def plot_multiple(plot_fn, count, max_columns=4, sharey='all'):
    rows = math.ceil(count / max_columns)
    fig, axes = plt.subplots(rows, min(count, max_columns), sharey=sharey, figsize=(6, rows * 1.8))

    for counter in range(count):
        if count == 1:
            ax = axes
        else:
            ax = axes.reshape(-1)[counter]

        plot_fn(counter, ax)

    for counter in range(count, rows * min(count, max_columns)):
        ax = axes.reshape(-1)[counter]
        ax.set_axis_off()

    fig.subplots_adjust(hspace=0.45)

    return fig