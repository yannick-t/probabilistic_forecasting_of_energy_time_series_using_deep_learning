import os
from datetime import datetime

import pandas
from sklearn.externals import joblib

from util.data.data_tools import preprocess_load_data_forec

'''
Utility methods to load, clean, select relevant information, and save clean data for load forecasting data
'''


def load_uci_load():
    load_processed_path = '../tmp/uci_load_15_clean.csv'

    if not os.path.exists(load_processed_path):
        dataset = pandas.read_csv('../../LD2011_2014.txt', header=0, low_memory=False,
                                  infer_datetime_format=True, parse_dates=[0],
                                  index_col=0, sep=';', decimal=',')


        load_df = pandas.DataFrame(dataset.sum(axis=1), columns=['load'])
        # shorten df so data for most clients is available
        load_df = load_df.loc[datetime(year=2012, month=1, day=1):, ['load']]

        # drop nan at end and beginning
        first_idx = load_df.first_valid_index()
        last_idx = load_df.last_valid_index()
        load_df = load_df.loc[first_idx:last_idx]
        # interpolate rest
        if load_df.isna().any()[0]:
            load_df = load_df.interpolate()

        load_df.to_csv(load_processed_path)
    else:
        load_df = pandas.read_csv(load_processed_path, header=0,
                                  infer_datetime_format=True, parse_dates=[0], index_col=[0])
    return load_df


def load_opsd_de_load_statistics():
    load_processed_path = '../tmp/opsd_de_load_statistics_60_clean.csv'
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
    load_processed_path = '../tmp/opsd_de_load_transparency_15_clean.csv'
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


def load_opsd_de_load_dataset(type, short_term=True, reprocess=False, scaler=None, n_ahead=1):
    assert type == 'transparency' or type == 'statistics'
    assert (scaler is not None and reprocess) or (scaler is None)

    if not os.path.exists('../tmp/'):
        os.mkdir('../tmp/')

    if type == 'transparency':
        load_fn = load_opsd_de_load_transparency
        dataset_path = '../tmp/de_load_transparency_'
        scaler_path = '../tmp/de_load_transparency_scaler.save'
    elif type == 'statistics':
        load_fn = load_opsd_de_load_statistics
        dataset_path = '../tmp/de_load_statistics_'
        scaler_path = '../tmp/de_load_statistics_scaler.save'

    dataset_path = dataset_path + 'n_ahead%d_' % n_ahead

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
            train_df, test_df, scaler = preprocess_load_data_forec(dataset, short_term=short_term, quarter_hour=False, scaler=scaler, n_ahead=n_ahead)
        else:
            train_df, test_df, scaler = preprocess_load_data_forec(dataset, short_term=short_term, quarter_hour=True, scaler=scaler, n_ahead=n_ahead)

        train_df.to_csv(dataset_path + 'train.csv')
        test_df.to_csv(dataset_path + 'test.csv')

        joblib.dump(scaler, scaler_path)

    return train_df, test_df, scaler
