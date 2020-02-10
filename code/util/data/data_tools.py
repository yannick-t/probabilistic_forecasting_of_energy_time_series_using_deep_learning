from datetime import timedelta

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from workalendar.europe import Germany


def convert_data_overlap(data, n, num_y=1, y_as_nx1=True):
    cal = Germany()

    x_arr = []
    y_arr = []
    for i in range(0, data.size - n):
        if (i + n + 1 + num_y) < data.size:
            # last n values
            x = data.iloc[i:i + n, 0].values
            x = np.append(x, (1 if data.iloc[i + n + 1].name.isoweekday() > 5 else 0))
            # holiday encoding
            x = np.append(x, 1 if cal.is_holiday(data.iloc[i + n + 1].name) else 0)

            x_arr.append(x)
            if y_as_nx1 and num_y == 1:
                y_arr.append([data.iloc[i + n + 1, 0]])
            else:
                y_arr.append(data.iloc[i + n + 1:i + n + 1 + num_y, 0])

    return np.array(x_arr), np.array(y_arr)


def preprocess_load_data_forec(dataframe):
    # pre process, extract features for forecasting
    cal = Germany()

    # use GW for convenience and readability later, also the standard-scaled values are smaller
    dataframe = dataframe / 1000
    # de-seasonalize days to some degree by substracting load values from same day last week
    dataframe_old = dataframe.copy()
    dataframe_old.index = dataframe.index.shift(freq=timedelta(days=7), periods=1)
    offset = dataframe_old[dataframe_old.index[0]: dataframe_old.index[-1] - timedelta(days=7)]
    dataframe = dataframe[dataframe.index[0] + timedelta(days=7): dataframe.index[-1]] - offset
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

    # day encoding weekday
    dataset_x_arrays.append(np.array([1 if time.isoweekday() > 5 else 0 for time in dataframe_adj.index]))

    # day encoding special days similar to
    # Arora, S., & Taylor, J. W. (2018).
    # Rule-based autoregressive moving average models for forecasting load on special days: A case study for France.
    # European Journal of Operational Research, 266(1), 259-268.
    dataset_x_arrays.append(np.array([1 if cal.is_holiday(time) else 0 for time in dataframe_adj.index]))
    # special day adjacent days
    dataset_x_arrays.append(np.array([1 if
                                cal.is_holiday(time + timedelta(days=1)) or cal.is_holiday(time - timedelta(days=1))
                                else 0 for time in dataframe_adj.index]))
    # bridging proximity days i.e., friday and monday
    dataset_x_arrays.append(np.array([1 if
                                (cal.is_holiday(time + timedelta(days=1)) or cal.is_holiday(time - timedelta(days=1))) and
                                (time.isoweekday == 1 or time.isoweekday == 5)
                                else 0 for time in dataframe_adj.index]))

    # lagged variables
    # values for same time n-days before  TODO: optimize this
    dataset_x_arrays.append(np.array([dataframe.loc[time - timedelta(days=1)] for time in dataframe_adj.index]).squeeze())
    dataset_x_arrays.append(np.array([dataframe.loc[time - timedelta(days=2)] for time in dataframe_adj.index]).squeeze())
    dataset_x_arrays.append(np.array([dataframe.loc[time - timedelta(days=3)] for time in dataframe_adj.index]).squeeze())
    dataset_x_arrays.append(np.array([dataframe.loc[time - timedelta(days=4)] for time in dataframe_adj.index]).squeeze())
    dataset_x_arrays.append(np.array([dataframe.loc[time - timedelta(days=5)] for time in dataframe_adj.index]).squeeze())
    # value for same time week before
    dataset_x_arrays.append(np.array([dataframe.loc[time - timedelta(days=7)] for time in dataframe_adj.index]).squeeze())

    dataset_x = np.stack(dataset_x_arrays, axis=-1)
    assert dataset_x.shape == (dataframe_adj.size, 14)

    return dataset_x, dataset_y, scaler, np.array(offset)


def inverse_transform_normal(mean, std, scaler):
    # utility method to inverse transform normal distribution parameters
    mean_transformed = scaler.inverse_transform(mean)
    mean_std_transformed = scaler.inverse_transform(mean + std)
    std_transformed = np.abs(mean_std_transformed - mean_transformed)

    return mean_transformed, std_transformed