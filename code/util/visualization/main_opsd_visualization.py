import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from util.data.data_src_tools import load_opsd_de_load_transparency

'''
Code to visualize the Opsd data used for forecasting.
'''


def main():

    dataset = load_opsd_de_load_transparency()
    load = dataset['load']
    # sns.kdeplot(load)
    # plt.show()

    load = load.resample('4D')
    load = load.mean()
    load = load.dropna()
    load = load.to_frame()

    load = load / 1000

    fig = plt.figure(figsize=(6, 3))

    ax = plt.subplot(1, 1, 1)
    ax.margins(0, 0.06)

    ax.set_ylabel('GW')

    ax.plot(load, label='Load Germany avg.', linewidth=1)
    # ax.plot(load_test, label='Load Germany avg.', linewidth=1)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17, left=0.09, right=0.91)
    plt.show()

    load_train, load_test = train_test_split(load, test_size=0.2, shuffle=False)
    fig = plt.figure(figsize=(6, 3))

    ax = plt.subplot(1, 1, 1)
    ax.margins(0, 0.06)

    ax.set_ylabel('GW')

    ax.plot(load_train, label='Training Data', linewidth=1)
    ax.plot(load_test, label='Test Data', linewidth=1)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17, left=0.09, right=0.91)
    plt.show()


main()
