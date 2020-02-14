import time
import numpy as np
from util.data.data_tools import inverse_transform_normal
import pandas
import datetime


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
