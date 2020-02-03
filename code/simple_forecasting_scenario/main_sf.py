import os

import matplotlib.pyplot as plt
import torch
import numpy as np

from evaluation.calibration import probabilistic_calibration, interval_coverage, marginal_calibration, \
    probabilistic_calibration_multiple, marginal_calibration_multiple
from evaluation.scoring import rmse, mape, crps, log_likelihood
from evaluation.sharpness import sharpness_plot_multiple, sharpness_plot, sharpness_avg_width
from models.concrete_dropout import ConcreteDropoutNN
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from training.loss.crps_loss import CRPSLoss
from training.loss.heteroscedastic_loss import HeteroscedasticLoss
from util.data.data_src_tools import load_opsd_de_load_daily, prepare_opsd_daily
from util.data.data_tools import convert_data_overlap, inverse_transform_normal
from util.visualization.plotting import default_plt_style

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

num_prev_val = 7
num_pred_val = 1

model_folder = './code/trained_models/'
model_prefix = 'simple_forecasting_'


def main():
    x_full, y_full, x_train, y_train, x_test, y_test, scaler = prepare_opsd_daily(num_prev_val, num_pred_val)
    model = concrete_init(x_train, y_train)
    load_train(model, x_train, y_train, 'concrete', load_saved=False)

    pred_y = model.predict(x_test)
    pred_y_full = model.predict(x_full)

    pred_y_mean = pred_y[..., 0]
    pred_y_var = pred_y[..., 1]
    pred_y_epis_var = pred_y[..., 2]**2
    pred_y_aleo_var = pred_y[..., 3]**2

    pred_y_full_mean = pred_y_full[..., 0]
    pred_y_full_var = pred_y_full[..., 1]

    pred_y_mean, pred_y_std = inverse_transform_normal(pred_y_mean, np.sqrt(pred_y_var), scaler)
    pred_y_var = pred_y_std**2
    y_test = scaler.inverse_transform(y_test)

    evaluate(pred_y_mean, pred_y_var, y_test)


def concrete_init(x_train, y_train):
    concrete_model = ConcreteSkorch(module=ConcreteDropoutNN,
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

    if os.path.isfile(model_file) & load_saved:
        model.initialize()
        model.load_params(model_file)
    else:
        model.fit(x_train, y_train)
        model.save_params(model_file)


def evaluate(pred_mean, pred_var, true_y):
    default_plt_style(plt)

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
    ax.set_title('Sharpness: Predictive Interval Width')
    sharpness_plot(pred_var, ax,)
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
