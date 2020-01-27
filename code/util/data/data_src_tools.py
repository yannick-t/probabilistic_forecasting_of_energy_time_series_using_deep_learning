from math import nan

import pandas
import torch
from numpy import float32
from sklearn.model_selection import train_test_split

from util.data.data_tools import convert_data_overlap


def process_opsd_de_load():
    dataset = pandas.read_csv('../time_series_60min_singleindex.csv', header=0, low_memory=False,
                       infer_datetime_format=True, parse_dates=['utc_timestamp'], index_col=['utc_timestamp'])

    daily_avg_power = dataset['DE_load_actual_entsoe_power_statistics']
    daily_avg_power = daily_avg_power.resample('D')
    daily_avg_power = daily_avg_power.mean()
    daily_avg_power = daily_avg_power.dropna()
    daily_avg_power = daily_avg_power.to_frame()

    daily_avg_power.to_csv('../DE_load_actual_entsoe_power_statistics_daily_avg.csv')


def load_opsd_de_load_daily():
    dataset = pandas.read_csv('../DE_load_actual_entsoe_power_statistics_daily_avg.csv', header=0,
                              infer_datetime_format=True, parse_dates=[0], index_col=[0])
    return dataset


def prepare_opsd_de_daily(device, num_prev_val=7, num_pred_val=1, y_as_nx1=True):
    # data
    dataset = load_opsd_de_load_daily()
    dataset_normalized = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    dataset_train, dataset_test = train_test_split(dataset_normalized, test_size=0.1, shuffle=False)

    # try to predict next value by last num_prev_val values
    x_train, y_train = convert_data_overlap(dataset_train, num_prev_val, num_y=num_pred_val, y_as_nx1=y_as_nx1)
    x_test, y_test = convert_data_overlap(dataset_test, num_prev_val, num_y=num_pred_val, y_as_nx1=y_as_nx1)

    x_train_tensor = torch.tensor(x_train, device=device).double()
    y_train_tensor = torch.tensor(y_train, device=device).double()
    x_test_tensor = torch.tensor(x_test, device=device).double()
    y_test_tensor = torch.tensor(y_test, device=device).double()

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor