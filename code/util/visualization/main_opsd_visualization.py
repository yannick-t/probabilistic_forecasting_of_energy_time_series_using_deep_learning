import matplotlib.pyplot as plt
import pandas


def main():
    fig = plt.figure()
    plt.style.use('seaborn-whitegrid')

    dataset = pandas.read_csv('../../../../time_series_60min_singleindex.csv', header=0, low_memory=False,
                              infer_datetime_format=True, parse_dates=['utc_timestamp'], index_col=['utc_timestamp'])

    daily_avg_power = dataset['DE_load_actual_entsoe_power_statistics']
    daily_avg_power = daily_avg_power.resample('14D')
    daily_avg_power = daily_avg_power.mean()
    daily_avg_power = daily_avg_power.dropna()
    daily_avg_power = daily_avg_power.to_frame()

    daily_avg_power = daily_avg_power / 1000

    ax = plt.subplot(1, 1, 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0, 0.06)

    ax.set_ylabel('GW')

    ax.plot(daily_avg_power, label='Load Germany avg.')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


main()
