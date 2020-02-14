from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import workalendar.europe.germany as wk
from statsmodels.tsa.api import ExponentialSmoothing, STL, seasonal_decompose
import matplotlib.pyplot as plt


def preprocess_load_data_forec(dataframe, quarter_hour=True, scaler=None):
    # use GW for convenience and readability later, also the standard-scaled values are smaller
    dataframe = dataframe / 1000

    # split data first so scaler and deseasonilizing can be trained on train set properly
    train_df_o, test_df_o = train_test_split(dataframe, test_size=0.2, shuffle=False)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(np.array(train_df_o['load']).reshape(-1, 1))
    train_df = pd.DataFrame({'load': scaler.transform(np.array(train_df_o['load']).reshape(-1, 1)).squeeze()},
                           index=train_df_o.index)
    test_df = pd.DataFrame({'load': scaler.transform(np.array(test_df_o['load']).reshape(-1, 1)).squeeze()},
                            index=test_df_o.index)

    # deseasonalize
    offset_train = pd.DataFrame(0, index=train_df.index, columns=['load'])
    offset_test = pd.DataFrame(0, index=test_df.index, columns=['load'])
    # decomp and train Holt Winters on decomp
    seasonal_periods = [24, 24 * 7]
    freq = 'H'

    if quarter_hour:
        seasonal_periods = [p * 4 for p in seasonal_periods]
        freq = '15T'

    for p in seasonal_periods:
        decomp = seasonal_decompose(train_df, period=p)
        exp = ExponentialSmoothing(decomp.seasonal, seasonal_periods=p, seasonal='add', freq=freq).fit()

        train_pred = exp.predict(start=train_df.index[0], end=train_df.index[-1])
        test_pred = exp.predict(start=test_df.index[0], end=test_df.index[-1])
        train_df['load'] = (train_df['load'] - train_pred)
        test_df['load'] = (test_df['load'] - test_pred)

        offset_train['load'] = offset_train['load'] + train_pred
        offset_test['load'] = offset_test['load'] + test_pred

    # construct features
    train_df = construct_features(train_df, offset_train, quarter_hour)
    test_df = construct_features(test_df, offset_test, quarter_hour)

    return train_df, test_df, scaler


def construct_features(dataframe, offset, quarter_hour=True):
    # pre process, define features for forecasting

    # calendars to be used for holiday encoding, complete Germany
    # as well as some important (high load) states
    cal = wk.Germany()
    calendars = [cal, wk.NorthRhineWestphalia(), wk.Bavaria(), wk.BadenWurttemberg(), wk.LowerSaxony()]

    # adjust for lagged variables so there are lagged variables for all targets
    adj_days = 7
    dataframe_adj = dataframe[dataframe.index[0] + timedelta(days=adj_days): dataframe.index[-1]]
    offset = offset[offset.index[0] + timedelta(days=adj_days): offset.index[-1]]

    result = pd.DataFrame(index=dataframe_adj.index)
    result = result.assign(target=dataframe_adj['load'])
    result = result.assign(offset=offset['load'])

    # time of year, value between 0 and 1, high in summer, low in winter
    year_day = np.array([time.timetuple().tm_yday - 1 for time in dataframe_adj.index])
    result = result.assign(sin_yearday=(np.sin(((year_day / 365) * 2 * np.pi) - np.pi / 2) + 1) * 3)
    result = result.assign(cos_yearday=(np.cos(((year_day / 365) * 2 * np.pi) - np.pi / 2) + 1) * 3)
    # (diff. of time of year sin)
    # time of day, value between 0 and 1, high in mid day, low at night
    day_hour = np.array([time.timetuple().tm_hour for time in dataframe_adj.index])
    day_hour = day_hour + np.array([time.timetuple().tm_min / 60 for time in dataframe_adj.index])
    result = result.assign(sin_dayhour=(np.sin(((day_hour / 24) * 2 * np.pi) - np.pi / 2) + 1) * 2)
    result = result.assign(cos_dayhour=(np.cos(((day_hour / 24) * 2 * np.pi) - np.pi / 2) + 1) * 2)

    # day encoding weekend
    result = result.assign(weekday_oh=np.array([1 if time.isoweekday() <= 5 else 0 for time in dataframe_adj.index]))
    # day encoding day of week one hot
    for d in range(1, 8):
        result = result.assign(**{'day_of_week_oh_%d' % d: np.array([1 if time.isoweekday() == d else 0 for time in dataframe_adj.index])})

    # day encoding special days similar to
    # Arora, S., & Taylor, J. W. (2018).
    # Rule-based autoregressive moving average models for forecasting load on special days: A case study for France.
    # European Journal of Operational Research, 266(1), 259-268.
    for c in calendars:
        result = result.assign(
            **{'cal_%s_special_day' % c.name: np.array([1 if c.is_holiday(time) else 0 for time in dataframe_adj.index])})
        # special day (double) bridging days
        result = result.assign(
            **{'cal_%s_special_day_bridg_1' % c.name:
                   np.array([1 if
                             ((c.is_holiday(
                                 time + timedelta(days=1)) and time.isoweekday() == 1)
                              or
                              (c.is_holiday(
                                  time - timedelta(days=1)) and time.isoweekday() == 5))
                             else 0 for time in dataframe_adj.index])})
        result = result.assign(
            **{'cal_%s_special_day_bridg_1_double' % c.name:
                   np.array([1 if
                             ((c.is_holiday(time + timedelta(days=1)) and time.isoweekday() == 2)
                              or
                              (c.is_holiday(time - timedelta(days=1)) and time.isoweekday() == 4))
                             else 0 for time in dataframe_adj.index])})
        result = result.assign(
            **{'cal_%s_special_day_bridg_2_double' % c.name:
                   np.array([1 if
                             ((c.is_holiday(time + timedelta(days=2)) and time.isoweekday() == 1)
                              or
                              (c.is_holiday(time - timedelta(days=2)) and time.isoweekday() == 5))
                             else 0 for time in dataframe_adj.index])})

    # encode time between christmas and new years because it is usually anomalous
    result = result.assign(christmas_time=np.array([1 if
                                           (time.date().month == 12 and 24 <= time.date().day <= 31)
                                           else 0 for time in dataframe_adj.index]))
    result = result.assign(newyear_time=np.array(np.array([1 if
                                                  (time.date().month == 1 and 1 <= time.date().day <= 8)
                                                  else 0 for time in dataframe_adj.index])))

    # lagged variables
    # values for same time n-days before
    lagged_days = [-1, -2, -3, -4, -5, -7]
    for d in lagged_days:
        result = result.assign(**{'lagged_day_%d' % d: np.array(
                                          dataframe[dataframe.index[0] + timedelta(days=adj_days + d): dataframe.index[-1] + timedelta(days=d)]
                                      ).squeeze()})
        assert not (result['target'] == result['lagged_day_%d' % d]).sum() > 50

    # values for 3h before
    lagged_hours = [-1, -2, -3]
    for h in lagged_hours:
        result = result.assign(**{'lagged_hours_%d' % h: np.array(
            dataframe[dataframe.index[0] + timedelta(days=adj_days, hours=h): dataframe.index[-1] + timedelta(hours=h)]
        ).squeeze()})
        # sanity check to make sure the load to predict is not fed into the model
        # (some of the values are the same in the datasets used, probably because of filled
        # missing values, so allow some leeway)
        assert not (result['target'] == result['lagged_hours_%d' % h]).sum() > 50

    lagged_quarter_hours = []
    if quarter_hour:
        lagged_quarter_hours = [-15, -30, -45]
        for q in lagged_quarter_hours:
            result = result.assign(**{'lagged_quarter_hours_%d' % q: np.array(
                dataframe[
                dataframe.index[0] + timedelta(days=adj_days, minutes=q): dataframe.index[-1] + timedelta(minutes=q)]
            ).squeeze()})
            # sanity check to make sure the load to predict is not fed into the model
            assert not (result['target'] == result['lagged_hours_%d' % h]).sum() > 50

    assert result.shape == (
        dataframe_adj.shape[0],
        16 + len(calendars) * 4 + len(lagged_days) + len(lagged_hours) + len(lagged_quarter_hours))

    return result


def inverse_transform_normal(mean, std, scaler):
    # utility method to inverse transform normal distribution parameters
    mean_transformed = scaler.inverse_transform(mean)
    mean_std_transformed = scaler.inverse_transform(mean + std)
    std_transformed = np.abs(mean_std_transformed - mean_transformed)

    return mean_transformed, std_transformed