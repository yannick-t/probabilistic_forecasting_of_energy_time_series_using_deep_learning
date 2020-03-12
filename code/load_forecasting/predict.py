import time
import numpy as np
import pandas as pd
from datetime import timedelta
import statsmodels.api as sm

from util.data.data_tools import inverse_transform_normal, inverse_transform_normal_var


'''
Utility methods to make predictions for all the models at the same time.
'''


def predict_transform_multiple(models, x_test, scaler, offset_test=None):
    pred_means = []
    pred_vars = []
    pred_vars_aleo = []
    pred_vars_epis = []
    times = []
    for name in models:
        pmean, pvar, pvara, pvare, time = predict_transform(models[name], x_test, scaler, offset_test, name)
        pred_means.append(pmean)
        pred_vars.append(pvar)
        pred_vars_aleo.append(pvara)
        pred_vars_epis.append(pvare)
        times.append(time)

    return pred_means, pred_vars, pred_vars_aleo, pred_vars_epis, times


def predict_transform(model, x_test, scaler, offset_test=None, model_name=''):
    # predict and inverse transform
    pred_y_mean, pred_y_var, pred_y_var_aleo, pred_y_var_epis, pred_time = predict(model, x_test, model_name)

    pred_y_mean, pred_y_var = inverse_transform_normal_var(pred_y_mean, pred_y_var, scaler)
    if pred_y_var_aleo is not None:
        _, pred_y_var_aleo = inverse_transform_normal_var(pred_y_mean, pred_y_var_aleo, scaler)
    if pred_y_var_epis is not None:
        _, pred_y_var_epis = inverse_transform_normal_var(pred_y_mean, pred_y_var_epis, scaler)
    if offset_test is not None:
        pred_y_mean = pred_y_mean + offset_test

    return pred_y_mean, pred_y_var, pred_y_var_aleo, pred_y_var_epis, pred_time


def predict(model, x, model_name=''):
    if model_name == 'linear_reg':
        start = time.time_ns()
        pred = model.get_prediction(sm.add_constant(x))
        end = time.time_ns()

        pred_y_mean = pred.predicted_mean
        pred_y_var = pred.var_pred_mean
        pred_y_var_aleo = None
        pred_y_var_epis = None

        if len(pred_y_mean.shape) == 1:
            pred_y_mean = pred_y_mean.reshape(-1, 1)
            pred_y_var = pred_y_var.reshape(-1, 1)
    elif model_name == 'quantile_reg':
        start = time.time_ns()
        preds = [m.get_prediction(sm.add_constant(x)) for m in model]
        end = time.time_ns()

        pred_means = np.array([pred.predicted_mean for pred in preds])
        pred_means_var = np.array([pred.var_pred_mean for pred in preds])
        pred_y_mean = np.mean(pred_means, axis=0)

        pred_y_var_aleo = ((pred_means[2] - pred_means[0]) / 1.35) ** 2
        pred_y_var_epis = np.mean(pred_means_var, axis=0)

        pred_y_var = np.mean((pred_means ** 2), axis=0) - pred_y_mean ** 2 + np.mean((pred_means_var ** 2), axis=0)

        if len(pred_y_mean.shape) == 1:
            pred_y_mean = pred_y_mean.reshape(-1, 1)
            pred_y_var = pred_y_var.reshape(-1, 1)
            pred_y_var_aleo = pred_y_var_aleo.reshape(-1, 1)
            pred_y_var_epis = pred_y_var_epis.reshape(-1, 1)
    else:
        start = time.time_ns()
        pred_y = model.predict(x)
        end = time.time_ns()

        pred_y_mean = pred_y[..., 0]
        pred_y_var = pred_y[..., 1]
        if pred_y.shape[-1] == 4:
            pred_y_var_epis = pred_y[..., 2]
            pred_y_var_aleo = pred_y[..., 3]
        else:
            pred_y_var_epis = None
            pred_y_var_aleo = None

    print('predict time ' + model_name + ' %d ns' % (end - start))

    return pred_y_mean, pred_y_var, pred_y_var_aleo, pred_y_var_epis, (end - start)


def predict_multi_step(model, test_df, lagged_short_term, horizon=1440):
    assert horizon % 15 == 0
    # multi step forecast by feeding the predictions back into the model
    # for the short term lagged variables
    test_df_x = test_df.drop(columns=['target', 'offset'])

    start_delta_range = pd.timedelta_range(timedelta(minutes=0), timedelta(minutes=horizon), freq='15T', closed='left')
    # collect predictions in dataframe
    pred_multi = pd.DataFrame(np.nan, index=test_df_x.index,
                              columns=['pred%d' % sd.seconds for sd in start_delta_range])

    # itterate over start time and predict until horizon for each possible start time
    for start_delta in start_delta_range:
        # starting timestamp at which to predict, predict next 24h iteratively
        pred_timestamp_start = pd.date_range(test_df_x.index[0] + start_delta, test_df_x.index[-1]
                                             - timedelta(days=1), freq=pd.DateOffset(minutes=horizon))
        pred_column = 'pred%d' % start_delta.seconds

        for delta in pd.timedelta_range(timedelta(minutes=0), timedelta(minutes=horizon), freq='15T', closed='left'):
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

    return pred_multi



