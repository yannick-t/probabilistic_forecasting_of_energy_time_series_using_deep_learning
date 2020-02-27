import gpytorch
import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor

from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch


# wrapper for simple aleatoric uncertainty nn
# trained with Heteroscedastic or CRPS Loss
class SimpleGPSkorch(BaseNNSkorch):
    def __init__(self, x_train, y_train, device, *args, **kwargs):
        self.module__train_x = to_tensor(x_train, device)
        self.module__train_y = to_tensor(y_train, device)
        kwargs['device'] = device
        super().__init__(criterion=None, *args, **kwargs)

    def predict(self, X):
        self.module_.eval()
        self.module__likelihood.eval()

        pred = self.module__likelihood(self.module_(X))
        return np.stack([to_numpy(pred.mean), to_numpy(pred.variance)], -1)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        if self.mll is None:
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.module__likelihood, self.module_)
        y_true = to_tensor(y_true, device=self.device)
        return -self.mll(y_pred, y_true)

    def train_step_single(self, Xi, yi, **fit_params):
        self.module__likelihood.train()
        super(SimpleGPSkorch, self).train_step_single(Xi, yi, **fit_params)

    def evaluation_step(self, Xi, training=False):
        self.check_is_fitted()
        with torch.set_grad_enabled(training):
            self.module_.train(training)
            self.module__likelihood.train(training)
            return self.infer(Xi)

    def initialize_criterion(self):
        return self
