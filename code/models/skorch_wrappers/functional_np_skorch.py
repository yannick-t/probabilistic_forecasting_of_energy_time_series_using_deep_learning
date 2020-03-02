import torch
from skorch.utils import to_tensor, to_numpy
from torch import nn
import numpy as np
from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch


# helper class to wrap the fnp implementation using skorch
# taking into account reference set for fnp
class RegressionFNPSkorch(BaseNNSkorch):
    def __init__(self, train_size, reference_set_size_ratio=0.1, seed=None, *args, **kwargs):
        self.train_size = train_size
        self.reference_set_size_ratio = reference_set_size_ratio
        self.seed = seed  # seed for random choice of reference set to reproduce same choice if needed
        if 'module__num_M' not in kwargs:
            self.calc_params()
        
        kwargs['criterion'] = IdentityLoss
        super().__init__(*args, **kwargs)
        
    def calc_params(self):
        self.module__num_M = self.train_size - int(self.reference_set_size_ratio * self.train_size)

    def initialize_module(self):
        self.calc_params()
        super(RegressionFNPSkorch, self).initialize_module()

    def choose_r(self, X, y):
        if self.seed is not None:
            np.random.seed(self.seed)

        idx = np.arange(X.shape[0])
        idx_r = np.random.choice(idx, size=(int(self.reference_set_size_ratio * X.shape[0]),), replace=False)
        self.xR, self.yR = to_tensor(X[idx_r], device=self.device), to_tensor(y[idx_r], device=self.device)

        return idx_r

    def predict(self, X, samples=30):
        self.module_.eval()

        output_size = self.module__dim_y
        samples = 30
        dxy = np.zeros((X.shape[0], samples, output_size))

        with torch.no_grad():
            for j in range(samples):
                dxy[:, j] = to_numpy(self.module_.predict(to_tensor(X, device=self.device), self.xR, self.yR))

        mean, var = dxy.mean(axis=1), dxy.var(axis=1)

        r = np.stack([mean, var], -1)

        return r

    def infer(self, x, **fit_params):
        x = to_tensor(x, device=self.device)
        if isinstance(x, dict):
            x_dict = self._merge_x_and_fit_params(x, fit_params)
            # set reference sest
            x_dict['XR'] = self.xR
            x_dict['yR'] = self.yR
            return self.module_(**x_dict)
        return self.module_(x, XR=self.xR, yR=self.yR, **fit_params)

    def fit(self, X, y=None, **fit_params):
        # reference set and m
        idx_r = self.choose_r(X, y)
        idx_m = np.array([i for i in np.arange(X.shape[0]) if i not in idx_r])
        x_m, y_m = X[idx_m], y[idx_m]
        m_dict = {'XM': x_m, 'yM': y_m}
        
        super(RegressionFNPSkorch, self).fit(m_dict, y_m, **fit_params)


# simple Helper loss because the fnp implementation already returns built in loss on forward pass
class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, preds, target):
        return preds