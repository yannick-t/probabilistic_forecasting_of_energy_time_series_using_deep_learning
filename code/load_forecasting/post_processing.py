from scipy.optimize import minimize
import numpy as np

from evaluation.scoring import crps

'''
Simple post-processing recalibration step and utility class to model the recalibration for all methods.
'''


def recalibrate(pred_means, pred_vars, y_true):
    # correct over- / underdispersion and bias in a post processing step
    # minimize two variables over crps: x[0] adds to mean to correct bias, x[1] is a factor
    # predictive variances are multiplied with to correct for over / underdispersion

    def fn(x):
        pmn = pred_means + x[0]
        pvn = pred_vars * x[1]
        res = crps(pmn, np.sqrt(pvn), y_true)
        return res

    min_res = minimize(fn, x0=np.array([0, 1]), bounds=[(None, None), (0.001, None)], method='SLSQP', options={'eps': 1e-5})

    return Recal(min_res.x)


class Recal():
    def __init__(self, x):
        self.x = x

    def __call__(self, pred_means, pred_vars, *args, **kwargs):
        res = pred_means + self.x[0], pred_vars * self.x[1]
        return res
