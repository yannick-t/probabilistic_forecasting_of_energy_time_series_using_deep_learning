import time

import pandas as pd
import statsmodels.api as sm
import numpy as np
import torch
from skorch.callbacks import EarlyStopping, EpochScoring

from evaluation.evaluate_forecasting_util import evaluate_multiple, evaluate_single
from evaluation.scoring import crps
from load_forecasting.forecast_util import dataset_df_to_np
from load_forecasting.predict import predict_transform_multiple, predict_transform
from models.concrete_dropout import ConcreteDropoutNN
from models.deep_ensemble_sklearn import DeepEnsemble
from models.deep_gp import DeepGaussianProcess
from models.functional_np import RegressionFNP
from models.quantile_regression import QuantRegFix
from models.simple_nn import SimpleNN
from models.skorch_wrappers.aleotoric_nn_skorch import AleatoricNNSkorch
from models.skorch_wrappers.bnn_skorch import BNNSkorch
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from models.skorch_wrappers.deep_gp_skorch import DeepGPSkorch
from models.skorch_wrappers.functional_np_skorch import RegressionFNPSkorch
from models.torch_bnn import TorchBNN
from training.loss.crps_loss import CRPSLoss
from training.loss.heteroscedastic_loss import HeteroscedasticLoss
from training.loss.torch_loss_fns import crps_torch, bnll
from training.training_util import load_train
from util.data.data_src_tools import load_opsd_de_load_dataset
from util.data.data_tools import gen_synth_ood_data_like
from util.model_enum import ModelEnum

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    # trained model location and prefix
    model_folder = '../trained_models/'
    prefix = 'load_forecasting_'
    result_folder = '../results/'

    models = [ModelEnum.quantile_reg, ModelEnum.simple_nn_aleo, ModelEnum.concrete, ModelEnum.fnp,
               ModelEnum.deep_ens, ModelEnum.bnn, ModelEnum.dgp]
    # models = [ModelEnum.quantile_reg, ModelEnum.linear_reg]

    # Forecasting case with short term lagged vars
    evaluate_models(model_folder, prefix, result_folder, short_term=True, models=models, load_saved_models=False,
                    generate_plots=False)
    # Forecasting case without short term lagged vars
    evaluate_models(model_folder, prefix, result_folder, short_term=False, models=models, load_saved_models=False,
                    generate_plots=False)


def evaluate_models(model_folder, prefix, result_folder, short_term, models=None, load_saved_models=False,
                    crps_loss=False, generate_plots=True):
    # evaluate given models fot the thesis, training with found hyper parameters
    # (initialization with parameters is located in the init method for each method)
    # default to all models

    # prefix for models and results to be saved as
    if short_term:
        prefix = prefix + 'short_term_'
    if crps_loss:
        prefix = prefix + 'crps_'
    model_prefix = prefix
    if models is not None:
        result_prefix = prefix + '_'.join([m.name for m in models]) + '_'
    else:
        result_prefix = prefix + 'all_'

    # load / preprocess dataset if needed
    train_df, test_df, scaler = load_opsd_de_load_dataset('transparency', short_term=short_term, reprocess=False,
                                                          n_ahead=1)

    x_train, y_train, offset_train = dataset_df_to_np(train_df)
    x_test, y_test, offset_test = dataset_df_to_np(test_df)
    timestamp_test = test_df.index.to_numpy()

    y_test_orig = scaler.inverse_transform(y_test) + offset_test
    y_train_orig = scaler.inverse_transform(y_train) + offset_train

    # random data to serve as out of distribution data
    # to test epistemic unc. capabilities of models
    x_oods = []
    # similar data to test set in dimensions of load and real indicator values
    ood_0_df = gen_synth_ood_data_like(test_df, short_term=short_term, seed=322)
    x_ood_0, _, _ = dataset_df_to_np(ood_0_df)
    x_oods.append(x_ood_0)
    # very differenct ranges
    ood_1_df = gen_synth_ood_data_like(test_df, short_term=short_term, seed=42, variation=4)
    x_ood_1, _, _ = dataset_df_to_np(ood_1_df)
    x_oods.append(x_ood_1)
    # completely random (including indicator variables)
    np.random.seed(492)
    x_ood_2 = np.random.uniform(-2, 2, size=x_test.shape)
    x_oods.append(x_ood_2)

    # initialize and evaluate all methods, train and save if load_saved_models is false
    result_df = init_train_eval_all(x_train, y_train, x_test, offset_test, y_test_orig, x_oods, scaler,
                                    model_folder, model_prefix, result_prefix, result_folder, short_term,
                                    load_saved=load_saved_models, model_names=models, crps=crps_loss,
                                    generate_plots=generate_plots)

    # save result csv
    if not load_saved_models:
        # save train time
        result_df[['train_time']].to_csv(result_folder + result_prefix + 'train_time.csv')
    else:
        # load train time
        train_time_df = pd.read_csv(result_folder + result_prefix + 'train_time.csv', index_col=0)
        result_df.loc[:, 'train_time'] = train_time_df.loc[:, 'train_time']
    result_df.to_csv(result_folder + result_prefix + 'results.csv', index_label='method')


def init_train_eval_all(x_train, y_train, x_test, offset_test, y_test_orig, x_oods, scaler, model_folder,
                        model_prefix, result_prefix, result_folder, short_term, load_saved=False, model_names=None,
                        crps=False, generate_plots=True):
    if model_names is None:
        model_names = [n for n in ModelEnum]
    models = {}

    # init selected models
    for n in model_names:
        if n == ModelEnum.linear_reg:
            models[n.name] = linear_regression_init(x_train, y_train, short_term)
        elif n == ModelEnum.quantile_reg:
            models[n.name] = quantile_regression_init(x_train, y_train, short_term)
        elif n == ModelEnum.simple_nn_aleo:
            models[n.name] = simple_aleo_nn_init(x_train, y_train, short_term, crps)
        elif n == ModelEnum.concrete:
            models[n.name] = concrete_init(x_train, y_train, short_term, crps)
        elif n == ModelEnum.fnp:
            models[n.name] = fnp_init(x_train, y_train, short_term)
        elif n == ModelEnum.deep_ens:
            models[n.name] = deep_ensemble_init(x_train, y_train, short_term, crps)
        elif n == ModelEnum.bnn:
            models[n.name] = bnn_init(x_train, y_train, short_term, crps)
        elif n == ModelEnum.dgp:
            models[n.name] = deep_gp_init(x_train, y_train, short_term, crps)

    time_df = pd.DataFrame(index=models.keys(), columns=['train_time', 'predict_time'])

    # train or load initialized models
    for key in models:
        if key == ModelEnum.linear_reg.name:
            # not a skorch model, separate training
            start = time.time_ns()
            models[key] = models[key].fit()
            end = time.time_ns()
            time_df.loc[key, 'train_time'] = end - start
        elif key == ModelEnum.quantile_reg.name:
            # not a skorch model, separate training
            start = time.time_ns()
            models[key] = [models[key].fit(q=q) for q in [0.25, 0.5, 0.75]]
            end = time.time_ns()
            time_df.loc[key, 'train_time'] = end - start
        else:
            time_df.loc[key, 'train_time'] = load_train(models[key], x_train, y_train, key, model_folder, model_prefix,
                                                        load_saved=load_saved)

    # set up reference set in case the model was loaded (needed for fnp)
    if ModelEnum.fnp.name in models:
        models[ModelEnum.fnp.name].choose_r(x_train, y_train)

    pred_means, pred_vars, pred_vars_aleo, _, pred_times = predict_transform_multiple(models, x_test, scaler, offset_test=offset_test)
    time_df.loc[:, 'predict_time'] = pred_times

    pred_ood_vars = []
    for x_ood in x_oods:
        _, pred, _, _, _ = predict_transform_multiple(models, x_ood, scaler)
        pred_ood_vars.append(pred)

    # convert times
    time_df.loc[:, 'train_time'] = time_df.loc[:, 'train_time'] / (1e9 * 60)  # nanosecods to minutes
    time_df.loc[:, 'predict_time'] = time_df.loc[:, 'predict_time'] / 1e9  # nanosecods to seconds

    # evaluate models (including plots and scores)
    scores_df = evaluate_multiple(models.keys(), pred_means, pred_vars, pred_vars_aleo, y_test_orig, pred_ood_vars, result_folder, result_prefix,
                                  generate_plots=generate_plots)

    return pd.concat([time_df, scores_df], axis=1)


def init_train_eval_single(init_fn, x_train, y_train, x_test, offset_test, y_test_orig, x_ood, offset_ood, scaler, model_folder,
                           model_prefix, name, short_term):
    model = init_fn(x_train, y_train, short_term)
    load_train(model, x_train, y_train, name, model_folder, model_prefix, load_saved=True)
    pred_mean, pred_var, _ = predict_transform(model, x_test, scaler, offset_test, name)
    _, pred_ood_var, _ = predict_transform(model, x_ood, scaler, offset_ood, name)

    evaluate_single(pred_mean, pred_var, y_test_orig, pred_ood_var)


def linear_regression_init(x_train, y_train, short_term):
    # statsmodels linear regression as reference
    # different to other models because not scikit learn / skorch
    x_train_lr = sm.add_constant(x_train)
    linear_reg = sm.OLS(y_train, x_train_lr)

    return linear_reg


def quantile_regression_init(x_train, y_train, short_term):
    # statsmodels quantile regression as reference
    # different to other models because not scikit learn / skorch
    x_train_qr = sm.add_constant(x_train)
    quantile_reg = QuantRegFix(y_train, x_train_qr)

    return quantile_reg


def simple_aleo_nn_init(x_train, y_train, short_term, crps_loss=False):
    if crps_loss:
        loss = CRPSLoss
    else:
        loss = HeteroscedasticLoss

    if short_term:
        hs = [24, 64, 32]
        lr = 0.0005
        epochs = 3500
    else:
        hs = [132, 77, 50]
        lr = 5e-05
        epochs = 1253

    es = EarlyStopping(patience=75)
    simple_nn = AleatoricNNSkorch(
        module=SimpleNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=hs,
        lr=lr,
        batch_size=1024,
        max_epochs=epochs,
        train_split=None,
        optimizer=torch.optim.Adam,
        criterion=loss,
        device=device,
        verbose=1,
        # callbacks=[es]
    )

    return simple_nn


def deep_gp_init(x_train, y_train, short_term, crps_loss=False):
    if crps_loss:
        loss = crps_torch
    else:
        loss = bnll

    if short_term:
        epochs = 600
        lr = 0.0015
        hs = [4]
    else:
        epochs = 600
        lr = 0.0015
        hs = [4]

    dgp = DeepGPSkorch(
        module=DeepGaussianProcess,
        module__input_size=x_train.shape[-1],
        module__hidden_size=hs,
        module__output_size=y_train.shape[-1] * 2,
        module__num_inducing=128,
        lr=lr,
        max_epochs=epochs,
        batch_size=1024,
        train_split=None,
        verbose=1,
        base_loss=loss,
        optimizer=torch.optim.Adam,
        num_data=x_train.shape[0],
        device=device)

    return dgp


def bnn_init(x_train, y_train, short_term, crps_loss=False):
    if crps_loss:
        loss = CRPSLoss
    else:
        loss = HeteroscedasticLoss

    # paramters found through hyperparameter optimization
    if short_term:
        hs = [24, 64, 32]
        prior_mu = 0
        prior_sigma = 0.1
        lr = 0.000182
        epochs = 3306
    else:
        hs = [132, 77, 50]
        prior_mu = -5
        prior_sigma = 0.1
        lr = 0.000423
        epochs = 3430

    bnn = BNNSkorch(
        module=TorchBNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=hs,
        module__prior_mu=prior_mu,
        module__prior_sigma=prior_sigma,
        sample_count=30,
        lr=lr,
        max_epochs=epochs,
        train_split=None,
        verbose=1,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=loss,
        device=device)

    return bnn


def fnp_init(x_train, y_train, short_term):
    # Early Stopping when crps does not improve
    # (fnp can get worse if trained too long)
    def scoring(model, X, y):
        y_pred = model.predict(X.X['XM'], samples=5)
        score = crps(y_pred[..., 0], np.sqrt(y_pred[..., 1]), y)
        return score

    scorer = EpochScoring(scoring, on_train=True, name='crps')
    es = EarlyStopping(monitor='crps', patience=100)

    if short_term:
        epochs = 300
        hs_enc = [24, 64]
        hs_dec = [32]
        dim_u = 2
        dim_z = 32
        fb_z = 1.0
    else:
        epochs = 300
        hs_enc = [132, 77]
        hs_dec = [50]
        dim_u = 9
        dim_z = 32
        fb_z = 1.0

    fnp = RegressionFNPSkorch(
        module=RegressionFNP,
        module__dim_x=x_train.shape[-1],
        module__dim_y=y_train.shape[-1],
        module__hidden_size_enc=hs_enc,
        module__hidden_size_dec=hs_dec,
        optimizer=torch.optim.Adam,
        device=device,
        seed=42,
        module__dim_u=dim_u,
        module__dim_z=dim_z,
        module__fb_z=fb_z,
        lr=0.001,
        reference_set_size_ratio=0.003,
        max_epochs=epochs,
        batch_size=1024,
        train_size=x_train.size,
        train_split=None,
        verbose=1,
        # callbacks=[scorer, es]
    )

    return fnp


def deep_ensemble_init(x_train, y_train, short_term, crps_loss=False):
    if crps_loss:
        loss = CRPSLoss
    else:
        loss = HeteroscedasticLoss

    # paramters found through hyperparameter optimization
    if short_term:
        hs = [24, 64, 32]
        lr = 0.0005
        epochs = 3500
    else:
        hs = [132, 77, 50]
        lr = 5.026324529403299e-05
        epochs = 1253

    ensemble_model = DeepEnsemble(
        input_size=x_train.shape[-1],
        output_size=y_train.shape[-1] * 2,
        hidden_size=hs,
        lr=lr,
        max_epochs=epochs,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=loss,
        device=device
    )

    return ensemble_model


def concrete_init(x_train, y_train, short_term, crps_loss=False):
    if crps_loss:
        loss = CRPSLoss
    else:
        loss = HeteroscedasticLoss

    # paramters found through hyperparameter optimization
    if short_term:
        hs = [24, 64, 32]
        lr = 0.00051
        epochs = 3400
        lengthscale = 1e-9
    else:
        hs = [132, 77, 50]
        lr = 0.0001
        epochs = 271
        lengthscale = 1e-9

    concrete_model = ConcreteSkorch(
        module=ConcreteDropoutNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=hs,
        lengthscale=lengthscale,
        dataset_size=x_train.shape[0],
        sample_count=30,
        lr=lr,
        train_split=None,
        verbose=1,
        max_epochs=epochs,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=loss,
        device=device)

    return concrete_model


if __name__ == '__main__':
    main()
