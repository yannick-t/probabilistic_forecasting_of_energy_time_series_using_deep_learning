import torch
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from skorch.callbacks import EarlyStopping, EpochTimer, ProgressBar

from load_forecasting.forecast_util import dataset_df_to_np
from models.concrete_dropout import ConcreteDropoutNN
from models.deep_ensemble_sklearn import DeepEnsemble
from models.deep_gp import DeepGaussianProcess
from models.functional_np import RegressionFNP
from models.skorch_wrappers.aleotoric_nn_skorch import AleatoricNNSkorch
from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch
from models.simple_nn import SimpleNN
from models.skorch_wrappers.bnn_skorch import BNNSkorch
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from models.skorch_wrappers.deep_gp_skorch import DeepGPSkorch
from models.skorch_wrappers.functional_np_skorch import RegressionFNPSkorch
from models.torch_bnn import TorchBNN
from training.loss.heteroscedastic_loss import HeteroscedasticLoss
from util.data.data_src_tools import load_opsd_de_load_transparency, load_opsd_de_load_dataset
from hyperparameter_opt.bayesian_optimization import bayesian_optimization, mse_scorer, crps_scorer
from util.data.data_tools import preprocess_load_data_forec

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')


# benchmark using opsd data to make a simple forecast using different methods
# and hyperparameter optimization
def main():
    short_term = True
    train_df, test_df, scaler = load_opsd_de_load_dataset('transparency', short_term=short_term, reprocess=False)

    x_train, y_train, offset_train = dataset_df_to_np(train_df)
    x_test, y_test, offset_test = dataset_df_to_np(test_df)

    print('Bayesian Optimization')
    print('OPSD ENTSOE-E Transparency')
    print('short term: %r' % short_term)

    fnp_bo(x_train, y_train, x_test, y_test, short_term)


def deep_gp_bo(x_train, y_train, x_test, y_test):
    print('Deep GP')
    dgp = DeepGPSkorch(
        module=DeepGaussianProcess,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__num_inducing=128,
        max_epochs=40,
        batch_size=1024,
        train_split=None,
        optimizer=torch.optim.Adam,
        num_data=x_train.shape[0],
        device=device)

    space = {# 'lr': Real(0.001, 0.03, 'log-uniform'),
             'module__hidden_size_0': Integer(1, 4),
             # 'module__hidden_size_1': Integer(1, 8),
             # 'module__hidden_size_2': Integer(1, 4),
             # 'module__num_inducing': Integer(16, 512),
             }

    bayesian_optimization(dgp, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=4,
                          n_jobs=1, cv=3)  # workaround for pickling error of gpytorch stuff, can't run prallel


def bnn_bo(x_train, y_train, x_test, y_test, short_term):
    print('Bayesian Neural Network')
    if short_term:
        hs = [24, 64, 32]
        prior_mu = 0
        prior_sigma = 0.1
    else:
        hs = [132, 77, 50]
        prior_mu = 0
        prior_sigma = 0.1
    print(hs)
    bnn = BNNSkorch(
        module=TorchBNN,
        module__input_size=x_train.shape[-1],
        module__hidden_size=hs,
        module__output_size=y_train.shape[-1] * 2,
        module__prior_mu=prior_mu,
        module__prior_sigma=prior_sigma,
        sample_count=30,
        train_split=None,
        max_epochs=10,
        batch_size=1024,
        lr=0.001,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device,
        verbose=0
    )

    space = {
        'lr': Real(0.00005, 0.005, 'log-uniform'),
        'max_epochs': Integer(350, 4200),
        # 'module__hidden_size_0': Integer(16, 256),
        # 'module__hidden_size_1': Integer(16, 512),
        # 'module__hidden_size_2': Integer(1, 256),
        # 'module__hidden_size_3': Integer(1, 256),
        # 'module__prior_mu': Real(-8, 6),
        # 'module__prior_sigma': Real(0.001, 0.11),
             }

    bayesian_optimization(bnn, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=28, cv=3)


def deep_ens_bo(x_train, y_train, x_test, y_test, short_term):
    print('Deep Ensemble')
    if short_term:
        hs = [24, 64, 32]
    else:
        hs = [132, 77, 50]
    print(hs)
    deep_ens = DeepEnsemble(
        input_size=x_train.shape[-1],
        hidden_size=hs,
        output_size=y_train.shape[-1] * 2,
        lr=0.0015,
        max_epochs=30,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device,
        parallel=False,
        verbose=1
    )

    space = {
        'lr': Real(0.00005, 0.005, 'log-uniform'),
        'max_epochs': Integer(350, 3500),
        # 'lr': Real(0.01, 0.1, 'log-uniform'),
        # 'module__hidden_size_0': Integer(1, 256),
        # 'module__hidden_size_1': Integer(1, 256),
        # 'module__hidden_size_2': Integer(1, 256),
        # 'module__hidden_size_3': Integer(1, 256),
        # 'hidden_size_4': Integer(1, 1024),
        # 'hidden_size_5': Integer(1, 1024),
    }

    bayesian_optimization(deep_ens, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=28)


def fnp_bo(x_train, y_train, x_test, y_test, short_term):
    progress_bar_ep = ProgressBar()
    early_stopping = EarlyStopping(monitor='train_loss', patience=8)
    cv = 3
    if short_term:
        hs_enc = [24, 64]
        hs_dec = [32]
    else:
        hs_enc = [132, 77]
        hs_dec = [50]
    print(hs_enc)
    print(hs_dec)
    print('Functional Neural Processes')
    fnp = RegressionFNPSkorch(
        module=RegressionFNP,
        module__dim_x=x_train.shape[-1],
        module__dim_y=y_train.shape[-1],
        module__hidden_size_enc=hs_enc,
        module__hidden_size_dec=hs_dec,
        train_split=None,
        optimizer=torch.optim.Adam,
        device=device,
        train_size=int((1 - 1 / cv) * x_train.shape[0]),
        module__dim_u=3,
        module__dim_z=50,
        module__fb_z=2.0,
        lr=0.001,
        reference_set_size_ratio=0.08,
        max_epochs=8,
        batch_size=1024,
        verbose=1,
        callbacks=[progress_bar_ep, early_stopping]
    )

    space = {
        # 'lr': Real(0.001, 0.01, 'log-uniform'),
        # 'max_epochs': Integer(1, 100),
        # 'batch_size': [32, 64, 128],
        'module__dim_u': Integer(1, 12),
        'module__dim_z': Integer(8, 128),
        'module__fb_z': Real(0, 2.4),
        # 'reference_set_size_ratio': Real(0.01, 0.4),
        # 'module__hidden_size_enc_0': Integer(1, 256),
        # 'module__hidden_size_enc_1': Integer(1, 256),
        # 'module__hidden_size_dec_0': Integer(1, 256),
        # 'module__hidden_size_dec_1': Integer(1, 256),
    }

    bayesian_optimization(fnp, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=24, cv=cv, n_jobs=2)


def concrete_bo(x_train, y_train, x_test, y_test, short_term):
    cv = 3
    print('Concrete Dropout')
    # hs = [6, 10, 32]
    # hs = [128, 16, 56]
    if short_term:
        hs = [24, 64, 32]
        lr = 0.00051
        epochs = 3400
    else:
        hs = [132, 77, 50]
        lr = 0.0001
        epochs = 271
    print(hs)
    concrete = ConcreteSkorch(module=ConcreteDropoutNN,
                              module__input_size=x_train.shape[-1],
                              module__hidden_size=hs,
                              module__output_size=y_train.shape[-1] * 2,
                              lengthscale=1e-6,
                              dataset_size=int((1 - 1 / cv) * x_train.shape[0]),
                              sample_count=30,
                              train_split=None,
                              max_epochs=epochs,
                              lr=lr,
                              batch_size=1024,
                              optimizer=torch.optim.Adam,
                              criterion=HeteroscedasticLoss,
                              device=device,
                              verbose=0)

    space = {
        # 'lr': Real(0.00005, 0.005, 'log-uniform'),
        # 'max_epochs': Integer(250, 3500),
        'lengthscale': Real(1e-9, 0.1, 'log-uniform'),
        # 'module__hidden_size_0': Integer(4, 133),
        # 'module__hidden_size_1': Integer(8, 80),
        # 'module__hidden_size_2': Integer(1, 64),
    }
    bayesian_optimization(concrete, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=28, cv=cv)


def simple_nn_bo(x_train, y_train, x_test, y_test):
    print('Simple NN')
    # simle nn as comparison, optimize hyperparameters using bayesian optimization
    simple_nn = BaseNNSkorch(module=SimpleNN,
                             module__input_size=x_train.shape[-1],
                             module__output_size=y_train.shape[-1],
                             optimizer=torch.optim.Adam,
                             criterion=torch.nn.MSELoss,
                             device=device,
                             lr=0.0015,
                             max_epochs=300,
                             batch_size=1024,
                             train_split=None,
                             verbose=0)

    space = {
        # 'lr': Real(0.01, 0.1, 'log-uniform'),
        # 'max_epochs': Integer(25, 500),
        'module__hidden_size_0': Integer(4, 133),
        'module__hidden_size_1': Integer(8, 80),
        'module__hidden_size_2': Integer(1, 64),
        # 'module__dropout_prob': Real(0, 0.5)
    }

    bayesian_optimization(simple_nn, space, mse_scorer, x_train, y_train, x_test, y_test, n_iter=200)


def simple_nn_aleo_bo(x_train, y_train, x_test, y_test, short_term):
    print('Simple NN Aleo')
    if short_term:
        hs = [24, 64, 32]
    else:
        hs = [132, 77, 50]
    print(hs)
    simple_nn = AleatoricNNSkorch(
        module=SimpleNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__hidden_size=hs,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device,
        lr=0.001,
        # max_epochs=1000,
        batch_size=1024,
        train_split=None,
        verbose=0)

    space = {
        'lr': Real(0.00005, 0.005, 'log-uniform'),
        'max_epochs': Integer(250, 3500),
        # 'module__hidden_size_0': Integer(4, 133),
        # 'module__hidden_size_1': Integer(8, 80),
        # 'module__hidden_size_2': Integer(1, 64),
        # 'module__dropout_prob': Real(0, 0.3)
    }

    bayesian_optimization(simple_nn, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=28, cv=3)


if __name__ == '__main__':
    main()
