import torch
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from skorch.callbacks import EarlyStopping

from models.concrete_dropout import ConcreteDropoutNN
from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch
from models.simple_nn import SimpleNN
from models.skorch_wrappers.concrete_skorch import ConcreteSkorch
from training.loss.concrete_heteroscedastic_loss import ConcreteHeteroscedasticLoss
from util.data.data_src_tools import load_opsd_de_load_daily
from util.data.data_tools import convert_data_overlap
from hyperparameter_opt.bayesian_optimization import bayesian_optimization, mse_scorer, crps_scorer

use_cuda = True
use_cuda = use_cuda & torch.cuda.is_available()

device = torch.device('cuda' if use_cuda else 'cpu')

num_prev_val = 7
num_pred_val = 1

dataset = load_opsd_de_load_daily()
dataset_normalized = (dataset - dataset.min()) / (dataset.max() - dataset.min())
x_full, y_full = convert_data_overlap(dataset_normalized, num_prev_val, num_y=num_pred_val, y_as_nx1=True)
dataset_train, dataset_test = train_test_split(dataset_normalized, test_size=0.1, shuffle=False)

# predict next value by last num_prev_val values
x_train, y_train = convert_data_overlap(dataset_train, num_prev_val, num_y=num_pred_val, y_as_nx1=True)
x_test, y_test = convert_data_overlap(dataset_test, num_prev_val, num_y=num_pred_val, y_as_nx1=True)


# benchmark using opsd data to make a simple forecast using different methods
# and hyperparameter optimization
def main():
    concrete_bo()


def concrete_bo():
    cv = 6
    concrete = ConcreteSkorch(module=ConcreteDropoutNN,
                              module__input_size=x_train.shape[-1],
                              module__output_size=y_train.shape[-1] * 2,
                              lengthscale=1e-4,
                              dataset_size=int((1 - 1 / cv) * x_train.shape[0]),
                              sample_count=30,
                              lr=0.001,
                              max_epochs=1500,
                              batch_size=1024,
                              optimizer=torch.optim.Adam,
                              criterion=ConcreteHeteroscedasticLoss,
                              device=device,
                              verbose=0)

    space = {
        'lr': Real(0.001, 0.04, 'log-uniform'),
        # 'max_epochs': Integer(50, 1000),
        # 'batch_size': Integer(1000, 4000),
        'module__hidden_size_0': Integer(64, 128),
        'module__hidden_size_1': Integer(64, 128),
        'module__hidden_size_2': Integer(12, 128),
        'module__hidden_size_3': Integer(1, 128),
        # 'module__hidden_size_4': Integer(1, 1024),
        # 'module__hidden_size_5': Integer(1, 1024),
    }

    bayesian_optimization(concrete, space, crps_scorer, x_train, y_train, x_test, y_test, n_iter=512, cv=cv)


def simple_nn_bo():
    # simle nn as comparison, optimize hyperparameters using bayesian optimization
    es = EarlyStopping(patience=200)
    simple_nn = BaseNNSkorch(module=SimpleNN,
                             module__input_size=x_train.shape[-1],
                             module__output_size=y_train.shape[-1],
                             optimizer=torch.optim.Adam,
                             criterion=torch.nn.MSELoss,
                             device=device,
                             callbacks=[es],
                             verbose=1)

    space = {'lr': Real(0.01, 0.1, 'log-uniform'),
             'max_epochs': Integer(50, 1000),
             'batch_size': Integer(1000, 4000),
             'module__hidden_size_0': Integer(16, 1024),
             'module__hidden_size_1': Integer(16, 1024),
             'module__hidden_size_2': Integer(1, 1024),
             # 'module__hidden_size_3': Integer(1, 1024),
             # 'module__hidden_size_4': Integer(1, 1024),
             # 'module__hidden_size_5': Integer(1, 1024),
             }

    bayesian_optimization(simple_nn, space, mse_scorer, x_train, y_train, x_test, y_test)


main()
