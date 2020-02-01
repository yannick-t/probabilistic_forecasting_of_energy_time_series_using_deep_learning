import os

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split

from evaluation.calibration import probabilistic_calibration, interval_coverage, marginal_calibration
from evaluation.sharpness import sharpness
from models.concrete_dropout import ConcreteDropoutNN
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from training.loss.crps_loss import CRPSLoss
from training.loss.heteroscedastic_loss import HeteroscedasticLoss
from util.data.data_src_tools import prepare_opsd_de_daily, load_opsd_de_load_daily
from util.data.data_tools import convert_data_overlap
from util.visualization.plt_styling import default_plt_style

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

num_prev_val = 7
num_pred_val = 1

model_folder = './code/trained_models/'
model_prefix = 'simple_forecasting_'

dataset = load_opsd_de_load_daily()
dataset_normalized = (dataset - dataset.min()) / (dataset.max() - dataset.min())
x_full, y_full = convert_data_overlap(dataset_normalized, num_prev_val, num_y=num_pred_val, y_as_nx1=True)
dataset_train, dataset_test = train_test_split(dataset_normalized, test_size=0.1, shuffle=False)

# predict next value by last num_prev_val values
x_train, y_train = convert_data_overlap(dataset_train, num_prev_val, num_y=num_pred_val, y_as_nx1=True)
x_test, y_test = convert_data_overlap(dataset_test, num_prev_val, num_y=num_pred_val, y_as_nx1=True)


def main():
    concrete()


def concrete():
    load_saved = True
    model_name = 'concrete'

    concrete_model = ConcreteSkorch(module=ConcreteDropoutNN,
                                    module__input_size=x_train.shape[-1],
                                    module__output_size=y_train.shape[-1] * 2,
                                    module__hidden_size=[64, 64, 7],
                                    lengthscale=1e-4,
                                    dataset_size=x_train.shape[0],
                                    sample_count=30,
                                    lr=0.001,
                                    train_split=None,
                                    max_epochs=1000,
                                    batch_size=1024,
                                    optimizer=torch.optim.Adam,
                                    criterion=CRPSLoss,
                                    device=device,
                                    verbose=1)

    load_train_evaluate(concrete_model, model_name, load_saved)


def load_train_evaluate(model, model_name, load_saved):
    model_file = model_folder + model_prefix + model_name

    if os.path.isfile(model_file) & load_saved:
        model.initialize()
        model.load_params(model_file)
    else:
        model.fit(x_train, y_train)
        model.save_params(model_file)

    pred_y = model.predict(x_test)
    pred_y_full = model.predict(x_full)

    pred_y_mean = pred_y[..., 0]
    pred_y_var = pred_y[..., 1]

    pred_y_full_mean = pred_y_full[..., 0]
    pred_y_full_var = pred_y_full[..., 1]

    evaluate(pred_y_full_mean, pred_y_full_var, y_full)


def evaluate(pred_mean, pred_var, true_y):
    default_plt_style(plt)

    # calibration
    ax = plt.subplot(2, 2, 1)
    probabilistic_calibration(pred_mean, pred_var, true_y, ax)
    ax = plt.subplot(2, 2, 2)
    marginal_calibration(pred_mean, pred_var, true_y, ax)

    # interval coverage
    cov = interval_coverage(pred_mean, pred_var, true_y, 0.9)
    print('0.9 interval coverage: %.5f' % cov)

    cov = interval_coverage(pred_mean, pred_var, true_y, 0.5)
    print('0.5 interval coverage: %.5f' % cov)

    # sharpness
    ax = plt.subplot(2, 2, 3)
    avg_5, avg_9 = sharpness(pred_mean, pred_var, ax)
    print('Average central 50%% interval width: %.5f' % avg_5)
    print('Average central 90%% interval width: %.5f' % avg_9)


main()
