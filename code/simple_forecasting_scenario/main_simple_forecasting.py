import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skorch.callbacks import EarlyStopping

from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch
from models.simple_nn import SimpleNN
from util.data.data_src_tools import load_opsd_de_load_daily
from util.data.data_tools import convert_data_overlap

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
    simple_nn_bo()


def simple_nn_bo():
    global counter
    # simle nn as comparison, optimize hyperparameters using bayesian optimization
    es = EarlyStopping(patience=200)
    simple_nn = BaseNNSkorch(module=SimpleNN,
                             module__input_size=x_train.shape[-1],
                             module__output_size=y_train.shape[-1],
                             optimizer=torch.optim.Adam,
                             criterion=torch.nn.MSELoss,
                             device=device,
                             callbacks=[es],
                             verbose=0)

    def scorer(estimator, X, y):
        y_predicted = estimator.predict(X)
        return -np.sqrt(np.mean((y - y_predicted) ** 2))

    opt = BayesSearchCV(
        simple_nn,
        {'lr': Real(0.01, 0.1, 'log-uniform'),
         'max_epochs': Integer(50, 1000),
         'batch_size': Integer(1000, 4000),
         'module__hidden_size_0': Integer(16, 1024),
         'module__hidden_size_1': Integer(16, 1024),
         'module__hidden_size_2': Integer(1, 1024),
         # 'module__hidden_size_3': Integer(1, 1024),
         # 'module__hidden_size_4': Integer(1, 1024),
         # 'module__hidden_size_5': Integer(1, 1024),
         },
        scoring=scorer,
        n_iter=100,
        cv=3,
        verbose=10,
        n_jobs=3)

    counter = 0

    # callback handler
    def on_step(optim_result):
        global counter
        score = opt.best_score_
        print(opt.best_params_)
        print("best score: %s" % score)
        print('counter: %d' % counter)
        counter = counter + 1

    opt.fit(x_train, y_train, callback=on_step)

    print(opt.best_params_)
    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(x_test, y_test))


main()
