# Implentation of concepts from Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
# Simple and scalable predictive uncertainty estimation using deep ensembles.
# In Advances in neural information processing systems (pp. 6402-6413).
# using a custom sklean estimator to train an ensemble of neural nets
import multiprocessing

import sklearn
import numpy as np
import torch
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import RegressorMixin
from multiprocessing import Pool

from models.base_nn import hidden_size_extract
from models.deep_ensemble_single import DeepEnsembleSingle
from models.skorch_wrappers.combined_unc_nn_skorch import combine_uncertainties
from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch


class DeepEnsemble(sklearn.base.BaseEstimator, RegressorMixin):
    def __init__(self, input_size, output_size, optimizer, criterion, device='cpu',
                 batch_size=128, max_epochs=100, lr=0.001, ensemble_size=5, parallel=True, **kwargs):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.ensemble_size = ensemble_size
        self.parallel = parallel
        self.hidden_size = hidden_size_extract(kwargs, 'hidden_size')

        self.models = []
        self.init_ensemble()

    def initialize(self):
        for model in self.models:
            model.initialize()

    def set_params(self, **params):
        self.hidden_size = hidden_size_extract(params, 'module__hidden_size',
                                               delete_from_dict=True)  # for sklearn consistency
        super(DeepEnsemble, self).set_params(**params)
        self.init_ensemble()

    def init_ensemble(self):
        self.models = []
        for _ in range(self.ensemble_size):
            model = BaseNNSkorch(DeepEnsembleSingle, module__input_size=self.input_size, module__output_size=self.output_size,
                                 module__hidden_size=self.hidden_size,
                                 optimizer=self.optimizer, criterion=self.criterion, device=self.device, batch_size=self.batch_size,
                                 max_epochs=self.max_epochs, lr=self.lr,
                                 train_split=None, verbose=1)

            self.models.append(model)

    def fit(self, X, y):
        # fit each with shuffled copies of the training data separately
        # according to original paper
        args = [(model, X, y) for model in self.models]

        multiprocessing.set_start_method('spawn', force=True)

        if self.parallel:
            pool = Pool(self.ensemble_size)
            self.models = pool.starmap(self.fit_single, args)
        else:
            self.models = [self.fit_single(*a) for a in args]

        return self

    def fit_single(self, model, X, y):
        # shuffle
        assert X.shape[0] == y.shape[0]
        perm = np.random.permutation(X.shape[0])
        shuffled_X = X[perm]
        shuffled_y = y[perm]

        # fit
        model.fit(shuffled_X, shuffled_y)

        return model

    def predict(self, X):
        # Input validation
        X = check_array(X)

        # predict with each model
        output_size = self.output_size
        assert output_size % 2 == 0
        preds = torch.zeros((X.shape[0], self.ensemble_size, output_size))

        for j, model in enumerate(self.models):
            preds[:, j] = torch.as_tensor(model.predict(X), device=self.device)

        return combine_uncertainties(preds, output_size)

    def save_params(self, file):
        for i, model in enumerate(self.models):
            model.save_params(file + '_%d' % i)

    def load_params(self, file):
        for i, model in enumerate(self.models):
            model.load_params(file + '_%d' % i)
