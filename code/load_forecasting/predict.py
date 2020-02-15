import time
import numpy as np
from util.data.data_tools import inverse_transform_normal
import pandas as pd
from datetime import timedelta


def predict_transform_multiple(models, names, x_test, scaler):
    pred_means = []
    pred_vars = []
    times = []
    for name, model in zip(names, models):
        pmean, pvar, time = predict_transform(model, x_test, scaler, name)
        pred_means.append(pmean)
        pred_vars.append(pvar)
        times.append(time)

    return pred_means, pred_vars, times


def predict_transform(model, x_test, scaler, offset_test, model_name=''):
    # predict and inverse transform
    start = time.time_ns()
    pred_y = model.predict(x_test)
    end = time.time_ns()
    print('predict time ' + model_name + ' %d ns' % (end - start))

    pred_y_mean = pred_y[..., 0]
    pred_y_var = pred_y[..., 1]

    pred_y_mean, pred_y_std = inverse_transform_normal(pred_y_mean, np.sqrt(pred_y_var), scaler)
    pred_y_var = pred_y_std ** 2
    pred_y_mean = pred_y_mean + offset_test

    return pred_y_mean, pred_y_var, (end - start)


def predict_multi_step(model, test_df, lagged_short_term):
    # TODO: variable horizon
    # multi step forecast by feeding the predictions back into the model
    # for the short term lagged variables
    test_df_x = test_df.drop(columns=['target', 'offset'])

    start_delta_range = pd.timedelta_range(timedelta(minutes=0), timedelta(days=1), freq='15T', closed='left')
    # collect predictions in dataframe
    pred_multi = pd.DataFrame(np.nan, index=test_df_x.index,
                              columns=['pred%d' % sd.minutes for sd in start_delta_range])

    # itterate over start time and predict until horizon for each possible start time
    for start_delta in start_delta_range:
        # starting timestamp at which to predict, predict next 24h iteratively
        pred_timestamp_start = pd.date_range(test_df_x.index[0] + start_delta, test_df_x.index[-1]
                                             - timedelta(days=1), freq='D')
        pred_column = 'pred%d' % start_delta.minutes

        for delta in pd.timedelta_range(timedelta(minutes=0), timedelta(days=1), freq='15T', closed='left'):
            pred_timestamp = pred_timestamp_start + delta  # advance timestamp to predict at

            # select test data at timestamp
            x_test = test_df_x.loc[pred_timestamp].copy()
            # use previous predictions for short term lagged variables of input data
            # as far as possible
            for minutes in lagged_short_term:
                lagged = pred_multi.loc[pred_timestamp + timedelta(minutes=minutes), pred_column]
                if not lagged.isna().any():
                    # value available from previous prediction -> use TODO: sample if distribution
                    x_test.loc[pred_timestamp, 'lagged_short_term_%d' % minutes] = lagged.to_numpy()

            pred = model.predict(x_test.to_numpy())
            pred_multi.loc[pred_timestamp, pred_column] = pred.squeeze()
            print(pred_multi)

        print(pred_multi)

    return pred_multi.loc[:, 'pred'].dropna().to_numpy().reshape(-1, 1)



