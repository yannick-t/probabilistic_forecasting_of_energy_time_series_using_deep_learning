import datetime
import os

import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from skorch.callbacks import EarlyStopping

from evaluation.evaluate_forecasting_util import plot_test_data, evaluate_multiple, evaluate_multi_step
from evaluation.scoring import rmse, mape, crps, log_likelihood
from load_forecasting.predict import predict_transform_multiple, predict_multi_step
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
from training.training_util import load_train
from util.data.data_src_tools import load_opsd_de_load_statistics, load_opsd_de_load_transparency, load_opsd_de_load_dataset
from util.data.data_tools import inverse_transform_normal, preprocess_load_data_forec
import time

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

model_folder = './trained_models/'
model_prefix = 'load_forecasting_'


def main():
    dataset = load_opsd_de_load_transparency()
    train_df, test_df, scaler = load_opsd_de_load_dataset('transparency', short_term=True, reprocess=False)

    y_train, offset_train = train_df.loc[:, 'target'].to_numpy().reshape(-1, 1), train_df.loc[:, 'offset'].to_numpy().reshape(-1, 1)
    x_train = train_df.drop(columns=['target', 'offset']).to_numpy()
    y_test, offset_test = test_df.loc[:, 'target'].to_numpy().reshape(-1, 1), test_df.loc[:, 'offset'].to_numpy().reshape(-1, 1)
    x_test = test_df.drop(columns=['target', 'offset']).to_numpy()
    timestamp_test = test_df.index.to_numpy()

    np.random.seed(333)
    x_ood_rand = np.random.uniform(-3, 3, x_test.shape)
    y_test_orig = scaler.inverse_transform(y_test) + offset_test

    reg = simple_nn_init(x_train, y_train)
    train_time_simple = load_train(reg, x_train, y_train, 'simple_nn', model_folder=model_folder,
                                   model_prefix=model_prefix, load_saved=True)

    start = time.time_ns()
    pred = reg.predict(x_test)
    pred = scaler.inverse_transform(pred) + offset_test
    end = time.time_ns()
    assert pred.shape == (y_test.shape[0], 1)

    horizon = 1440  # horizon in minutes
    pred_multi = predict_multi_step(reg, test_df, lagged_short_term=[-60, -120, -180, -15, -30, -45], horizon=horizon)
    assert pred_multi.shape == (y_test.shape[0], horizon // 15)
    evaluate_multi_step(pred_multi, y_test_orig, offset_test, scaler)

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


def init_train_eval_all(x_train, y_train, x_test, y_test, y_test_orig, x_ood_rand, scaler):
    concrete = concrete_init(x_train, y_train)
    train_time_conc = load_train(concrete, x_train, y_train, 'concrete', load_saved=True)

    fnp = fnp_init(x_train, y_train)
    train_time_fnp = load_train(fnp, x_train, y_train, 'fnp', load_saved=True)
    fnp.choose_r(x_train, y_train)  # set up reference set in case the model was loaded

    deep_ens = deep_ensemble_init(x_train, y_train)
    train_time_deep_ens = load_train(deep_ens, x_train, y_train, 'deep_ens', load_saved=True)

    bnn = bnn_init(x_train, y_train)
    train_time_bnn = load_train(bnn, x_train, y_train, 'bnn', load_saved=True)

    dgp = deep_gp_init(x_train, y_train)
    train_time_deepgp = load_train(dgp, x_train, y_train, 'deep_gp', load_saved=True)

    names = ['Concrete', 'FNP', 'Deep Ens.', 'BNN', 'Deep GP']
    models = [concrete, fnp, deep_ens, bnn, dgp]

    pred_means, pred_vars, pred_times = predict_transform_multiple(models, names, x_test, scaler)
    _, pred_ood_vars, _ = predict_transform_multiple(models, names, x_ood_rand, scaler)


    print('train times: %d, %d, %d, %d, %d' % (
        train_time_conc, train_time_fnp, train_time_deep_ens, train_time_bnn, train_time_deepgp))
    print('pred times: %d, %d, %d, %d, %d' % (
        pred_times[0], pred_times[1], pred_times[2], pred_times[3], pred_times[4]))

    evaluate_multiple(names, pred_means, pred_vars, y_test_orig, pred_ood_vars)


def simple_nn_init(x_train, y_train):
    es = EarlyStopping(patience=75)
    simple_nn = BaseNNSkorch(
        module=SimpleNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1],
        module__hidden_size=[32, 16],
        lr=0.002,
        batch_size=2048,
        max_epochs=100,
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


if __name__ == '__main__':
    main()
