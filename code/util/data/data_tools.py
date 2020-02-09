from datetime import timedelta

import numpy as np
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

    dataframe = dataframe.dropna()
    # use GW for convenience and readability later, also the standard-scaled values are smaller
    dataframe = dataframe / 1000
    scaler = StandardScaler()
    dataframe['load'] = \
        scaler.fit_transform(np.array(dataframe['load']).reshape(-1, 1)).squeeze()

    dataset_x = np.zeros([dataframe.size, 5])
    dataset_y = np.zeros([dataframe.size, 1])

    dataset_y[:, 0] = np.array(dataframe['load'])

    # time of year, value between 0 and 1, high in summer, low in winter
    dataset_x[:, 0] = np.array([time.timetuple().tm_yday - 1 for time in dataframe.index])
    dataset_x[:, 0] = (np.sin(((dataset_x[:, 0] / 365) * 2 * np.pi) - np.pi / 2) + 1) * 3
    # time of day, value between 0 and 1, high in mid day, low at night
    dataset_x[:, 1] = np.array([time.timetuple().tm_hour for time in dataframe.index])
    dataset_x[:, 1] = dataset_x[:, 1] + np.array([time.timetuple().tm_min / 60 for time in dataframe.index])
    dataset_x[:, 1] = (np.sin(((dataset_x[:, 1] / 24) * 2 * np.pi) - np.pi / 2) + 1) * 2
    # day encoding weekday
    dataset_x[:, 2] = np.array([1 if time.isoweekday() > 5 else 0 for time in dataframe.index])
    # day encoding holidays TODO: special day adjacency (bridging etc.)
    dataset_x[:, 3] = np.array([1 if cal.is_holiday(time) else 0 for time in dataframe.index])
    # special day adjacent days
    dataset_x[:, 4] = np.array([1 if
                                cal.is_holiday(time + timedelta(days=1)) or cal.is_holiday(time - timedelta(days=1))
                                else 0 for time in dataframe.index])
    # lagged variables # TODO: lagged variables
    # same time day before

    return dataset_x, dataset_y, scaler


def inverse_transform_normal(mean, std, scaler):
    # utility method to inverse transform normal distribution parameters
    mean_transformed = scaler.inverse_transform(mean)
    mean_std_transformed = scaler.inverse_transform(mean + std)
    std_transformed = np.abs(mean_std_transformed - mean_transformed)

    return mean_transformed, std_transformed