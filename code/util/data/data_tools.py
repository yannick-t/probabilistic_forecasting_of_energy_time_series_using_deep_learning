import numpy as np
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


def inverse_transform_normal(mean, std, scaler):
    # utility method to inverse transform normal distribution parameters
    mean_transformed = scaler.inverse_transform(mean)
    mean_std_transformed = scaler.inverse_transform(mean + std)
    std_transformed = np.abs(mean_std_transformed - mean_transformed)

    return mean_transformed, std_transformed