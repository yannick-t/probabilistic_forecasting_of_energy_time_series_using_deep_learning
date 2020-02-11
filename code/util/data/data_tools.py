from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import workalendar.europe.germany as wk


def preprocess_load_data_forec(dataframe):
    # pre process, extract features for forecasting

    # calendars to be used for holiday encoding, complete Germany
    # as well as some important states
    cal = wk.Germany()
    calendars = [cal, wk.NorthRhineWestphalia(), wk.Bavaria(), wk.BadenWurttemberg(), wk.LowerSaxony()]

    # use GW for convenience and readability later, also the standard-scaled values are smaller
    dataframe = dataframe / 1000
    # de-seasonalize days to some degree by substracting load values from same day last week
    # dataframe_old = dataframe.copy()
    # dataframe = dataframe[dataframe.index[0] + timedelta(days=7): dataframe.index[-1]]

    offset = pd.DataFrame(0, index=dataframe.index, columns=['load'])

    # dataframe_old.index = dataframe_old.index.shift(freq=timedelta(days=7), periods=1)
    # offset = offset + dataframe_old[dataframe_old.index[0]: dataframe_old.index[-1] - timedelta(days=7)]

    dataframe = dataframe - offset
    # standard scale
    scaler = StandardScaler()
    dataframe['load'] = \
        scaler.fit_transform(np.array(dataframe['load']).reshape(-1, 1)).squeeze()

    # adjust for lagged variables so there are lagged variables for all targets
    dataframe_adj = dataframe[dataframe.index[0] + timedelta(days=7): dataframe.index[-1]]
    offset = offset[offset.index[0] + timedelta(days=7): offset.index[-1]]

    dataset_y = np.zeros([dataframe_adj.size, 1])
    dataset_y[:, 0] = np.array(dataframe_adj['load'])

    dataset_x_arrays = []
    # time of year, value between 0 and 1, high in summer, low in winter
    year_day = np.array([time.timetuple().tm_yday - 1 for time in dataframe_adj.index])
    dataset_x_arrays.append((np.sin(((year_day / 365) * 2 * np.pi) - np.pi / 2) + 1) * 3)
    dataset_x_arrays.append((np.cos(((year_day / 365) * 2 * np.pi) - np.pi / 2) + 1) * 3)  # diff. of time of year sin
    # time of day, value between 0 and 1, high in mid day, low at night
    day_hour = np.array([time.timetuple().tm_hour for time in dataframe_adj.index])
    day_hour = day_hour + np.array([time.timetuple().tm_min / 60 for time in dataframe_adj.index])
    dataset_x_arrays.append((np.sin(((day_hour / 24) * 2 * np.pi) - np.pi / 2) + 1) * 2)
    dataset_x_arrays.append((np.cos(((day_hour / 24) * 2 * np.pi) - np.pi / 2) + 1) * 2)  # diff. of time of day sin

    # day encoding weekend / weekday
    dataset_x_arrays.append(np.array([1 if time.isoweekday() > 5 else 0 for time in dataframe_adj.index]))
    # day encoding day of week one hot
    for d in range(1, 8):
        dataset_x_arrays.append(np.array([1 if time.isoweekday() == d else 0 for time in dataframe_adj.index]))

    # day encoding special days similar to
    # Arora, S., & Taylor, J. W. (2018).
    # Rule-based autoregressive moving average models for forecasting load on special days: A case study for France.
    # European Journal of Operational Research, 266(1), 259-268.
    for c in calendars:  # calendars for some high load states as well as complete Ger
        dataset_x_arrays.append(np.array([1 if c.is_holiday(time) else 0 for time in dataframe_adj.index]))
        # special day (double) bridging days
        dataset_x_arrays.append(np.array([1 if
                                          ((c.is_holiday(time + timedelta(days=1)) and time.isoweekday() == 1)
                                           or
                                           (c.is_holiday(time - timedelta(days=1)) and time.isoweekday() == 5))
                                          else 0 for time in dataframe_adj.index]))
        dataset_x_arrays.append(np.array([1 if
                                          ((c.is_holiday(time + timedelta(days=1)) and time.isoweekday() == 2)
                                           or
                                           (c.is_holiday(time - timedelta(days=1)) and time.isoweekday() == 4))
                                          else 0 for time in dataframe_adj.index]))
        dataset_x_arrays.append(np.array([1 if
                                          ((c.is_holiday(time + timedelta(days=2)) and time.isoweekday() == 1)
                                           or
                                           (c.is_holiday(time - timedelta(days=2)) and time.isoweekday() == 5))
                                          else 0 for time in dataframe_adj.index]))

    # encode time between christmas and new years because it is usually anomalous
    dataset_x_arrays.append(np.array([1 if
                                      (time.date().month == 12 and 24 <= time.date().day <= 31)
                                      else 0 for time in dataframe_adj.index]))
    dataset_x_arrays.append(np.array([1 if
                                      (time.date().month == 1 and 1 <= time.date().day <= 8)
                                      else 0 for time in dataframe_adj.index]))

    # lagged variables
    # values for same time n-days before
    lagged_days = [-1, -2, -3, -4, -5, -7]
    for d in lagged_days:
        dataset_x_arrays.append(np.array(
            dataframe[dataframe.index[0] + timedelta(days=7 + d): dataframe.index[-1] + timedelta(days=d)]
        ).squeeze())

    dataset_x = np.stack(dataset_x_arrays, axis=-1)
    assert dataset_x.shape == (dataframe_adj.size, 14 + len(lagged_days) + len(calendars) * 4)

    return dataset_x, dataset_y, scaler, np.array(offset), np.array(dataframe_adj.index)


def inverse_transform_normal(mean, std, scaler):
    # utility method to inverse transform normal distribution parameters
    mean_transformed = scaler.inverse_transform(mean)
    mean_std_transformed = scaler.inverse_transform(mean + std)
    std_transformed = np.abs(mean_std_transformed - mean_transformed)

    return mean_transformed, std_transformed