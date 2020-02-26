import matplotlib.pyplot as plt
import seaborn as sns

from util.data.data_src_tools import load_opsd_de_load_transparency


def main():
    fig = plt.figure()

    dataset = load_opsd_de_load_transparency()
    load = dataset['load']
    sns.kdeplot(load)
    plt.show()

    load = load.resample('4D')
    load = load.mean()
    load = load.dropna()
    load = load.to_frame()

    load = load / 1000

    ax = plt.subplot(1, 1, 1)
    ax.margins(0, 0.06)

    ax.set_ylabel('GW')

    ax.plot(load, label='Load Germany avg.')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


main()
