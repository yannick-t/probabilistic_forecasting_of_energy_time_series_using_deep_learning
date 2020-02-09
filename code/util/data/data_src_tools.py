import os

import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from util.data.data_tools import convert_data_overlap


def process_opsd_de_load_to_daily():
    dataset = pandas.read_csv('../time_series_60min_singleindex.csv', header=0, low_memory=False,
                       infer_datetime_format=True, parse_dates=['utc_timestamp'], index_col=['utc_timestamp'])

    daily_avg_power = dataset['DE_load_actual_entsoe_power_statistics']
    daily_avg_power = daily_avg_power.resample('D')
    daily_avg_power = daily_avg_power.mean()
    daily_avg_power = daily_avg_power.dropna()
    daily_avg_power = daily_avg_power.to_frame()

    daily_avg_power.to_csv('../DE_load_actual_entsoe_power_statistics_daily_avg.csv')


def load_opsd_de_load_daily():
    dataset = pandas.read_csv('../../DE_load_actual_entsoe_power_statistics_daily_avg.csv', header=0,
                              infer_datetime_format=True, parse_dates=[0], index_col=[0])
    return dataset


def load_opsd_de_load_statistics():
    load_processed_path = '../../opsd_de_load_statistics_60_clean.csv'
    if not os.path.exists(load_processed_path):
        dataset = pandas.read_csv('../../time_series_60min_singleindex.csv', header=0, low_memory=False,
                                  infer_datetime_format=True, parse_dates=['utc_timestamp'], index_col=['utc_timestamp'])
        dataset.rename(columns={'DE_load_actual_entsoe_power_statistics': 'load'}, inplace=True)
        load_de_statistic = dataset[['load']]
        load_de_statistic = load_de_statistic.dropna()
        load_de_statistic.to_csv(load_processed_path)
    else:
        load_de_statistic = pandas.read_csv(load_processed_path, header=0,
                                            infer_datetime_format=True, parse_dates=[0], index_col=[0])
    return load_de_statistic


def prepare_opsd_daily(num_prev_val, num_pred_val):
    dataset = load_opsd_de_load_daily()
    # use GW for convenience and readability later, also the standard-scaled values are smaller
    dataset = dataset / 1000
    scaler = StandardScaler()
    dataset['DE_load_actual_entsoe_power_statistics'] = \
        scaler.fit_transform(np.array(dataset['DE_load_actual_entsoe_power_statistics']).reshape(-1, 1)).squeeze()

    x_full, y_full = convert_data_overlap(dataset, num_prev_val, num_y=num_pred_val, y_as_nx1=True)
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.1, shuffle=False)

    # predict next value by last num_prev_val values
    x_train, y_train = convert_data_overlap(dataset_train, num_prev_val, num_y=num_pred_val, y_as_nx1=True)
    x_test, y_test = convert_data_overlap(dataset_test, num_prev_val, num_y=num_pred_val, y_as_nx1=True)

    return x_full, y_full, x_train, y_train, x_test, y_test, scaler