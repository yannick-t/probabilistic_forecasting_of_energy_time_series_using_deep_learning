import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from evaluation.calibration import probabilistic_calibration, interval_coverage, marginal_calibration, \
    probabilistic_calibration_multiple, marginal_calibration_multiple
from evaluation.scoring import rmse, mape, crps, log_likelihood
from evaluation.sharpness import sharpness_plot_multiple, sharpness_plot, sharpness_avg_width, sharpness_plot_histogram, \
    sharpness_plot_histogram_joint, sharpness_plot_histogram_joint_multiple
from models.concrete_dropout import ConcreteDropoutNN
from models.deep_ensemble_sklearn import DeepEnsemble
from models.functional_np import RegressionFNP
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from models.skorch_wrappers.functional_np_skorch import RegressionFNPSkorch
from training.loss.crps_loss import CRPSLoss
from training.loss.heteroscedastic_loss import HeteroscedasticLoss
from util.data.data_src_tools import load_opsd_de_load_daily, prepare_opsd_daily
from util.data.data_tools import convert_data_overlap, inverse_transform_normal

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

num_prev_val = 7
num_pred_val = 1

model_folder = './trained_models/'
model_prefix = 'simple_forecasting_'


def main():
    x_full, y_full, x_train, y_train, x_test, y_test, scaler = prepare_opsd_daily(num_prev_val, num_pred_val)
    np.random.seed(322)
    x_ood_rand = np.random.uniform(-7, 7, x_test.shape)
    y_test_orig = scaler.inverse_transform(y_test)

    concrete = concrete_init(x_train, y_train)
    load_train(concrete, x_train, y_train, 'concrete', load_saved=True)

    fnp = fnp_init(x_train, y_train)
    load_train(fnp, x_train, y_train, 'fnp', load_saved=True)
    fnp.choose_r(x_train, y_train)  # set up reference set in case the model was loaded

    deep_ens = deep_ensemble_init(x_train, y_train)
    load_train(deep_ens, x_train, y_train, 'deep_ens', load_saved=True)

    pred_means, pred_vars = predict_transform_multiple([concrete, fnp, deep_ens], x_test, scaler)
    _, pred_ood_vars = predict_transform_multiple([concrete, fnp, deep_ens], x_ood_rand, scaler)

    evaluate_multiple(['Concrete', 'FNP', 'Deep Ens.'], pred_means, pred_vars, y_test_orig, pred_ood_vars)


def fnp_init(x_train, y_train):
    fnp = RegressionFNPSkorch(
        module=RegressionFNP,
        module__dim_x=x_train.shape[-1],
        module__dim_y=y_train.shape[-1],
        module__hidden_size_enc=[20],
        module__hidden_size_dec=[20],
        module__dim_u=3,
        module__dim_z=50,
        module__fb_z=1.0,
        optimizer=torch.optim.Adam,
        device=device,
        seed=42,
        max_epochs=10,
        batch_size=64,
        reference_set_size_ratio=0.1,
        train_size=x_train.size)

    return fnp


def deep_ensemble_init(x_train, y_train):
    ensemble_model = DeepEnsemble(
        input_size=x_train.shape[-1],
        output_size=y_train.shape[-1] * 2,
        hidden_size=[32, 48, 7],
        lr=0.001,
        max_epochs=100,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device
    )

    return ensemble_model


def concrete_init(x_train, y_train):
    concrete_model = ConcreteSkorch(
        module=ConcreteDropoutNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=[64, 64, 7],
        lengthscale=1e-4,
        dataset_size=x_train.shape[0],
        sample_count=30,
        lr=0.001,
        train_split=None,
        max_epochs=1024,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=CRPSLoss,
        device=device,
        verbose=1)

    return concrete_model


def load_train(model, x_train, y_train, model_name, load_saved):
    model_file = model_folder + model_prefix + model_name

    if load_saved:
        model.initialize()
        model.load_params(model_file)
    else:
        model.fit(x_train, y_train)
        model.save_params(model_file)


def predict_transform_multiple(models, x_test, scaler):
    pred_means = []
    pred_vars = []
    for model in models:
        pmean, pvar = predict_transform(model, x_test, scaler)
        pred_means.append(pmean)
        pred_vars.append(pvar)

    return pred_means, pred_vars


def predict_transform(model, x_test, scaler):
    # predict and inverse transform
    pred_y = model.predict(x_test)

    pred_y_mean = pred_y[..., 0]
    pred_y_var = pred_y[..., 1]

    pred_y_mean, pred_y_std = inverse_transform_normal(pred_y_mean, np.sqrt(pred_y_var), scaler)
    pred_y_var = pred_y_std ** 2

    return pred_y_mean, pred_y_var


def evaluate_multiple(names, pred_means, pred_vars, true_y, pred_ood_vars):
    # calibration
    probabilistic_calibration_multiple(names, pred_means, pred_vars, true_y)
    marginal_calibration_multiple(names, pred_means, pred_vars, true_y)

    # sharpness
    sharpness_plot_multiple(names, pred_vars)

    # epistemic out of distribution evaluation
    sharpness_plot_histogram_joint_multiple(names, pred_vars, pred_ood_vars)

    # scoring etc.
    for name, pmean, pvar in zip(names, pred_means, pred_vars):
        print('#########################################')
        print('Model: ' + name)
        cov = interval_coverage(pmean, pvar, true_y, 0.9)
        print('0.9 interval coverage: %.5f' % cov)

        cov = interval_coverage(pmean, pvar, true_y, 0.5)
        print('0.5 interval coverage: %.5f' % cov)

        avg_5, avg_9 = sharpness_avg_width(pvar)
        print('Average central 50%% interval width: %.5f' % avg_5)
        print('Average central 90%% interval width: %.5f' % avg_9)

        print("RMSE: %.4f" % rmse(pmean, true_y))
        print("MAPE: %.2f" % mape(pmean, true_y))
        print("CRPS: %.4f" % crps(pmean, np.sqrt(pvar), true_y))
        print("Average LL: %.4f" % log_likelihood(pmean, np.sqrt(pvar), true_y))


def evaluate_single(pred_mean, pred_var, true_y):
    # calibration
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Probabilistic Calibration: Probability Integral Transform Histogram')
    probabilistic_calibration(pred_mean, pred_var, true_y, ax)
    ax = plt.subplot(2, 2, 2)
    ax.set_title('Marginal Calibration: Difference between empirical CDF and average predictive CDF')
    marginal_calibration(pred_mean, pred_var, true_y, ax)

    # interval coverage
    cov = interval_coverage(pred_mean, pred_var, true_y, 0.9)
    print('0.9 interval coverage: %.5f' % cov)

    cov = interval_coverage(pred_mean, pred_var, true_y, 0.5)
    print('0.5 interval coverage: %.5f' % cov)

    # sharpness
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Sharpness: Predictive Interval Width Boxplot')
    sharpness_plot(pred_var, ax)
    ax = plt.subplot(2, 2, 4)
    ax.set_title('Sharpness: Predictive Interval Width Histogram')
    sharpness_plot_histogram(pred_var, ax)
    avg_5, avg_9 = sharpness_avg_width(pred_var)
    print('Average central 50%% interval width: %.5f' % avg_5)
    print('Average central 90%% interval width: %.5f' % avg_9)

    # scoring
    print("RMSE: %.4f" % rmse(pred_mean, true_y))
    print("MAPE: %.2f" % mape(pred_mean, true_y))
    print("CRPS: %.4f" % crps(pred_mean, np.sqrt(pred_var), true_y))
    print("Average LL: %.4f" % log_likelihood(pred_mean, np.sqrt(pred_var), true_y))

    plt.show()


def plot_test_data(pred_mean, pred_var, y_true, ax):
    x = np.array(range(pred_mean.shape[0]))

    ax.plot(y_true.squeeze(), color='blue')
    ax.plot(pred_mean.squeeze(), color='orange')

    if pred_var is not None:
        std_deviations = np.sqrt(pred_var.squeeze())

        for j in range(1, 5):
            ax.fill_between(x, pred_mean.squeeze() - j / 2 * std_deviations,
                            pred_mean.squeeze() + j / 2 * std_deviations,
                            alpha=0.1, color='orange')


if __name__ == '__main__':
    main()
