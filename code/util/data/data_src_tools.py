import os

import pandas
import numpy as np
from sklearn.externals import joblib
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from util.data.data_tools import preprocess_load_data_forec


def load_opsd_de_load_statistics():
    load_processed_path = '../../opsd_de_load_statistics_60_clean.csv'
    if not os.path.exists(load_processed_path):
        dataset = pandas.read_csv('../../time_series_60min_singleindex.csv', header=0, low_memory=False,
                                  infer_datetime_format=True, parse_dates=['utc_timestamp'],
                                  index_col=['utc_timestamp'])
        dataset.rename(columns={'DE_load_actual_entsoe_power_statistics': 'load'}, inplace=True)
        load_de_statistic = dataset[['load']]
        load_de_statistic = load_de_statistic.dropna()
        load_de_statistic.to_csv(load_processed_path)
    else:
        load_de_statistic = pandas.read_csv(load_processed_path, header=0,
                                            infer_datetime_format=True, parse_dates=[0], index_col=[0])
    return load_de_statistic


def load_opsd_de_load_transparency():
    load_processed_path = '../../opsd_de_load_transparency_15_clean.csv'
    if not os.path.exists(load_processed_path):
        dataset = pandas.read_csv('../../time_series_15min_singleindex.csv', header=0, low_memory=False,
                                  infer_datetime_format=True, parse_dates=['utc_timestamp'],
                                  index_col=['utc_timestamp'])
        dataset.rename(columns={'DE_load_actual_entsoe_transparency': 'load'}, inplace=True)
        load_de_transparency = dataset[['load']]

        # drop nan at end and beginning
        first_idx = load_de_transparency.first_valid_index()
        last_idx = load_de_transparency.last_valid_index()
        load_de_transparency = load_de_transparency.loc[first_idx:last_idx]
        # interpolate rest
        if load_de_transparency.isna().any()[0]:
            load_de_transparency = load_de_transparency.interpolate()

        load_de_transparency.to_csv(load_processed_path)
    else:
        load_de_transparency = pandas.read_csv(load_processed_path, header=0,
                                               infer_datetime_format=True, parse_dates=[0], index_col=[0])
    return load_de_transparency


def load_opsd_de_load_dataset(type, short_term=True, reprocess=False, scaler=None):
    assert type == 'transparency' or type == 'statistics'
    assert (scaler is not None and reprocess) or (scaler is None)

    if type == 'transparency':
        load_fn = load_opsd_de_load_transparency
        dataset_path = '../de_load_transparency_'
        scaler_path = '../de_load_transparency_scaler.save'
    elif type == 'statistics':
        load_fn = load_opsd_de_load_statistics
        dataset_path = '../de_load_statistics_'
        scaler_path = '../de_load_statistics_scaler.save'

    if short_term:
        dataset_path = dataset_path + 'short_term_'

    if os.path.exists(dataset_path + 'train.csv') and not reprocess:
        print('loading saved dataset')

        train_df = pandas.read_csv(dataset_path + 'train.csv', infer_datetime_format=True,
                                   parse_dates=['utc_timestamp'], index_col=['utc_timestamp'])
        test_df = pandas.read_csv(dataset_path + 'test.csv', infer_datetime_format=True,
                                   parse_dates=['utc_timestamp'], index_col=['utc_timestamp'])

        scaler = joblib.load(scaler_path)
    else:
        # process dataset and save
        dataset = load_fn()
        if type == 'statistics':
            train_df, test_df, scaler = preprocess_load_data_forec(dataset, short_term=short_term, quarter_hour=False, scaler=scaler)
        else:
            train_df, test_df, scaler = preprocess_load_data_forec(dataset, short_term=short_term, quarter_hour=True, scaler=scaler)

        train_df.to_csv(dataset_path + 'train.csv')
        test_df.to_csv(dataset_path + 'test.csv')

        joblib.dump(scaler, scaler_path)

    return train_df, test_df, scaler
