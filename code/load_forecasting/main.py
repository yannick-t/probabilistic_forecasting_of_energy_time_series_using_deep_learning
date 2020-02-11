import datetime
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skorch.callbacks import EarlyStopping

from evaluation.calibration import probabilistic_calibration, interval_coverage, marginal_calibration, \
    probabilistic_calibration_multiple, marginal_calibration_multiple
from evaluation.scoring import rmse, mape, crps, log_likelihood
from evaluation.sharpness import sharpness_plot_multiple, sharpness_plot, sharpness_avg_width, sharpness_plot_histogram, \
    sharpness_plot_histogram_joint, sharpness_plot_histogram_joint_multiple
from models.concrete_dropout import ConcreteDropoutNN
from models.deep_ensemble_sklearn import DeepEnsemble
from models.deep_gp import DeepGaussianProcess
from models.functional_np import RegressionFNP
from models.simple_nn import SimpleNN
from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch
from models.skorch_wrappers.bnn_skorch import BNNSkorch
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from models.skorch_wrappers.deep_gp_skorch import DeepGPSkorch
from models.skorch_wrappers.functional_np_skorch import RegressionFNPSkorch
from models.torch_bnn import TorchBNN
from training.loss.crps_loss import CRPSLoss
from training.loss.heteroscedastic_loss import HeteroscedasticLoss
from util.data.data_src_tools import load_opsd_de_load_statistics, load_opsd_de_load_transparency, load_opsd_de_load_dataset
from util.data.data_tools import inverse_transform_normal, preprocess_load_data_forec
import time

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

model_folder = './trained_models/'
model_prefix = 'load_forecasting_'


def main():
    dataset_x, dataset_y, scaler, offset, timestamp = load_opsd_de_load_dataset(type='transparency', reprocess=True)
    x_train, x_test, y_train, y_test, offset_train, offset_test, timestamp_train, timestamp_test \
        = train_test_split(dataset_x, dataset_y, offset, timestamp,
                           test_size=0.2, shuffle=False)

    np.random.seed(333)
    x_ood_rand = np.random.uniform(-3, 3, x_test.shape)
    y_test_orig = scaler.inverse_transform(y_test) + offset_test

    reg = simple_nn_init(x_train, y_train)
    train_time_simple = load_train(reg, x_train, y_train, 'simple_nn', load_saved=False)

    # reg = LinearRegression().fit(x_train, y_train)

    # concrete = concrete_init(x_train, y_train)
    # train_time_conc = load_train(concrete, x_train, y_train, 'concrete', load_saved=True)
    #
    # fnp = fnp_init(x_train, y_train)
    # train_time_fnp = load_train(fnp, x_train, y_train, 'fnp', load_saved=True)
    # fnp.choose_r(x_train, y_train)  # set up reference set in case the model was loaded
    #
    # deep_ens = deep_ensemble_init(x_train, y_train)
    # train_time_deep_ens = load_train(deep_ens, x_train, y_train, 'deep_ens', load_saved=True)
    #
    # bnn = bnn_init(x_train, y_train)
    # train_time_bnn = load_train(bnn, x_train, y_train, 'bnn', load_saved=True)
    #
    # dgp = deep_gp_init(x_train, y_train)
    # train_time_deepgp = load_train(dgp, x_train, y_train, 'deep_gp', load_saved=True)
    #
    # names = ['Concrete', 'FNP', 'Deep Ens.', 'BNN', 'Deep GP']
    # models = [concrete, fnp, deep_ens, bnn, dgp]
    #
    # pred_means, pred_vars, pred_times = predict_transform_multiple(models, names, x_test, scaler)
    # _, pred_ood_vars, _ = predict_transform_multiple(models, names, x_ood_rand, scaler)
    #
    start = time.time_ns()
    pred = reg.predict(x_test)
    end = time.time_ns()
    pred = scaler.inverse_transform(pred) + offset_test
    #
    # print('train times: %d, %d, %d, %d, %d, %d' % (train_time_simple,
    #     train_time_conc, train_time_fnp, train_time_deep_ens, train_time_bnn, train_time_deepgp))
    # print('pred times: %d, %d, %d, %d, %d, %d' % (pred_time_simple,
    #     pred_times[0], pred_times[1], pred_times[2], pred_times[3], pred_times[4]))
    #
    # evaluate_multiple(names, pred_means, pred_vars, y_test_orig, pred_ood_vars)

    ax = plt.subplot(1, 1, 1)
    plot_test_data(pred, np.ones_like(pred) * 0.1, y_test_orig, timestamp_test, ax)
    # ax = plt.subplot(1, 2, 2)
    # pred_simple_train = reg.predict(x_train)
    # plot_test_data(pred_simple_train, np.ones_like(pred_simple_train) * 0.01, y_train, timestamp_train, ax)
    plt.show()

    print('###############################')
    print('Simple NN:')
    print("RMSE: %.4f" % rmse(pred, y_test_orig))
    print("MAPE: %.2f" % mape(pred, y_test_orig))


def simple_nn_init(x_train, y_train):
    es = EarlyStopping(patience=75)
    simple_nn = BaseNNSkorch(
        module=SimpleNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1],
        module__hidden_size=[32, 16],
        lr=0.002,
        batch_size=1024,
        max_epochs=200,
        train_split=None,
        optimizer=torch.optim.Adam,
        criterion=torch.nn.MSELoss,
        device=device,
        verbose=1,
        # callbacks=[es]
    )

    return simple_nn


def deep_gp_init(x_train, y_train):
    dgp = DeepGPSkorch(
        module=DeepGaussianProcess,
        module__input_size=x_train.shape[-1],
        module__hidden_size=[4],
        module__output_size=y_train.shape[-1] * 2,
        module__num_inducing=128,
        lr=0.01,
        max_epochs=150,
        batch_size=256,
        train_split=None,
        verbose=1,
        optimizer=torch.optim.Adam,
        num_data=x_train.shape[0],
        device=device)

    return dgp


def bnn_init(x_train, y_train):
    bnn = BNNSkorch(
        module=TorchBNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=[32, 48, 7],
        module__prior_mu=0,
        module__prior_sigma=0.1,
        sample_count=30,
        lr=0.001,
        max_epochs=4000,
        train_split=None,
        verbose=1,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device)

    return bnn


def fnp_init(x_train, y_train):
    fnp = RegressionFNPSkorch(
        module=RegressionFNP,
        module__dim_x=x_train.shape[-1],
        module__dim_y=y_train.shape[-1],
        module__hidden_size_enc=[64, 128, 19],
        module__hidden_size_dec=[128, 32],
        module__dim_u=12,
        module__dim_z=8,
        module__fb_z=1.0,
        optimizer=torch.optim.Adam,
        device=device,
        seed=42,
        max_epochs=64,
        batch_size=64,
        reference_set_size_ratio=0.1,
        train_size=x_train.size,
        train_split=None,
        verbose=1
    )

    return fnp


def deep_ensemble_init(x_train, y_train):
    ensemble_model = DeepEnsemble(
        input_size=x_train.shape[-1],
        output_size=y_train.shape[-1] * 2,
        hidden_size=[32, 48, 7],
        lr=0.001,
        max_epochs=3000,
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
        verbose=1,
        max_epochs=3000,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device)

    return concrete_model


def load_train(model, x_train, y_train, model_name, load_saved):
    model_file = model_folder + model_prefix + model_name

    if load_saved:
        model.initialize()
        model.load_params(model_file)

        return 0
    else:
        start = time.time_ns()
        model.fit(x_train, y_train)
        end = time.time_ns()
        print('fit time ' + model_name + ' %d ns' % (end - start))
        model.save_params(model_file)

        return end - start


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


def predict_transform(model, x_test, scaler, offset_test, model_name=""):
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


def plot_test_data(pred_mean, pred_var, y_true, timestamp, ax):
    x = timestamp

    ax.plot(x, y_true.squeeze(), color='blue')
    ax.plot(x, pred_mean.squeeze(), color='orange')

    ax.plot(x, (y_true.squeeze() - pred_mean.squeeze())**2, color='red')

    if pred_var is not None:
        std_deviations = np.sqrt(pred_var.squeeze())

        for j in range(1, 5):
            ax.fill_between(x, pred_mean.squeeze() - j / 2 * std_deviations,
                            pred_mean.squeeze() + j / 2 * std_deviations,
                            alpha=0.1, color='orange')


if __name__ == '__main__':
    main()
