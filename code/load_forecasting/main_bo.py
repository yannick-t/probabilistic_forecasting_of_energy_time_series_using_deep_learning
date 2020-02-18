import torch
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from skorch.callbacks import EarlyStopping

from models.concrete_dropout import ConcreteDropoutNN
from models.deep_ensemble_sklearn import DeepEnsemble
from models.deep_gp import DeepGaussianProcess
from models.functional_np import RegressionFNP
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
    short_term = False
    train_df, test_df, scaler = load_opsd_de_load_dataset('transparency', short_term=short_term, reprocess=False)

    y_train, offset_train = train_df.loc[:, 'target'].to_numpy().reshape(-1, 1), train_df.loc[:, 'offset'].to_numpy()\
        .reshape(-1, 1)
    x_train = train_df.drop(columns=['target', 'offset']).to_numpy()
    y_test, offset_test = test_df.loc[:, 'target'].to_numpy().reshape(-1, 1), test_df.loc[:, 'offset'].to_numpy()\
        .reshape(-1, 1)
    x_test = test_df.drop(columns=['target', 'offset']).to_numpy()

    print('Bayesian Optimization')
    print('OPSD ENTSOE-E Transparency')
    print('short term: %r' % short_term)

    simple_nn_bo(x_train, y_train, x_test, y_test)


def deep_gp_bo(x_train, y_train, x_test, y_test):
    dgp = DeepGPSkorch(
        module=DeepGaussianProcess,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__num_inducing=128,
        max_epochs=1000,
        batch_size=256,
        train_split=None,
        optimizer=torch.optim.Adam,
        num_data=x_train.shape[0],
        device=device)

    space = {# 'lr': Real(0.001, 0.03, 'log-uniform'),
             'module__hidden_size_0': Integer(1, 4),
             'module__hidden_size_1': Integer(1, 8),
             'module__hidden_size_2': Integer(1, 4),
             # 'module__num_inducing': Integer(16, 512),
             }

    bayesian_optimization(dgp, space, mse_scorer, x_train, y_train, x_test, y_test, n_iter=256,
                          n_jobs=1)  # workaround for pickling error of gpytorch stuff, can't run prallel


def bnn_bo(x_train, y_train, x_test, y_test):
    bnn = BNNSkorch(
        module=TorchBNN,
        module__input_size=x_train.shape[-1],
        module__output_size=y_train.shape[-1] * 2,
        module__prior_mu=0,
        module__prior_sigma=0.1,
        sample_count=30,
        train_split=None,
        max_epochs=5000,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device,
        verbose=0
    )

    space = {'lr': Real(0.001, 0.03, 'log-uniform'),
             'module__hidden_size_0': Integer(16, 256),
             'module__hidden_size_1': Integer(16, 512),
             'module__hidden_size_2': Integer(1, 256),
             'module__hidden_size_3': Integer(1, 256),
             'module__prior_mu': Real(-10, 10),
             'module__prior_sigma': Real(0.001, 0.15),
             # 'module__hidden_size_4': Integer(1, 1024),
             # 'module__hidden_size_5': Integer(1, 1024),
             }

    bayesian_optimization(bnn, space, mse_scorer, x_train, y_train, x_test, y_test, n_iter=512)


def deep_ens_bo(x_train, y_train, x_test, y_test):
    deep_ens = DeepEnsemble(
        input_size=x_train.shape[-1],
        output_size=y_train.shape[-1] * 2,
        lr=0.001,
        max_epochs=2000,
        batch_size=1024,
        optimizer=torch.optim.Adam,
        criterion=HeteroscedasticLoss,
        device=device
    )

    space = {'lr': Real(0.01, 0.1, 'log-uniform'),
             'hidden_size_0': Integer(16, 1024),
             'hidden_size_1': Integer(16, 1024),
             'hidden_size_2': Integer(1, 1024),
             'hidden_size_3': Integer(1, 1024),
             # 'hidden_size_4': Integer(1, 1024),
             # 'hidden_size_5': Integer(1, 1024),
             }

    bayesian_optimization(deep_ens, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=512)


def fnp_bo(x_train, y_train, x_test, y_test):
    cv = 5
    fnp = RegressionFNPSkorch(
        module=RegressionFNP,
        module__dim_x=x_train.shape[-1],
        module__dim_y=y_train.shape[-1],
        train_split=None,
        optimizer=torch.optim.Adam,
        device=device,
        train_size=int((1 - 1 / cv) * x_train.shape[0]),
        verbose=0
    )

    space = {
        'lr': Real(0.001, 0.01, 'log-uniform'),
        'max_epochs': Integer(1, 2),
        'batch_size': [32, 64, 128],
        'module__dim_u': Integer(1, 16),
        'module__dim_z': Integer(8, 128),
        'module__fb_z': Real(0, 2.0),
        'reference_set_size_ratio': Real(0.01, 0.4),
        'module__hidden_size_enc_0': Integer(64, 256),
        'module__hidden_size_enc_1': Integer(64, 512),
        'module__hidden_size_enc_2': Integer(1, 256),
        'module__hidden_size_dec_0': Integer(16, 256),
        'module__hidden_size_dec_1': Integer(16, 512),
        'module__hidden_size_dec_2': Integer(1, 256),
    }

    bayesian_optimization(fnp, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=1024, cv=cv)


def concrete_bo(x_train, y_train, x_test, y_test):
    cv = 5
    concrete = ConcreteSkorch(module=ConcreteDropoutNN,
                              module__input_size=x_train.shape[-1],
                              module__output_size=y_train.shape[-1] * 2,
                              lengthscale=1e-4,
                              dataset_size=int((1 - 1 / cv) * x_train.shape[0]),
                              sample_count=30,
                              lr=0.001,
                              train_split=None,
                              max_epochs=200,
                              batch_size=1024,
                              optimizer=torch.optim.Adam,
                              criterion=HeteroscedasticLoss,
                              device=device,
                              verbose=0)

    space = {
        # 'lr': Real(0.001, 0.04, 'log-uniform'),
        # 'lengthscale': Real(1e-8, 0.1, 'log-uniform'),
        # 'max_epochs': Integer(50, 1000),
        'module__hidden_size_0': Integer(1, 128),
        'module__hidden_size_1': Integer(1, 128),
        'module__hidden_size_2': Integer(1, 256),
        'module__hidden_size_3': Integer(1, 128),
    }
    bayesian_optimization(concrete, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=200, cv=cv)


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
                             max_epochs=108,
                             batch_size=1024,
                             train_split=None,
                             verbose=0)

    space = {
        # 'lr': Real(0.01, 0.1, 'log-uniform'),
        # 'max_epochs': Integer(25, 500),
        'module__hidden_size_0': Integer(1, 128),
        'module__hidden_size_1': Integer(1, 128),
        'module__hidden_size_2': Integer(1, 256),
        'module__hidden_size_3': Integer(1, 128),
        # 'module__dropout_prob': Real(0, 0.5)
    }

    bayesian_optimization(simple_nn, space, mse_scorer, x_train, y_train, x_test, y_test, n_iter=200)


main()
