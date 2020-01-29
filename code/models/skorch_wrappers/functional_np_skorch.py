import torch
from skorch.utils import to_tensor, to_numpy
from torch import nn
import numpy as np
from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch


# helper class to wrap the fnp implementation using skorch
# taking into account reference set for fnp
class RegressionFNPSkorch(BaseNNSkorch):
    def __init__(self, reference_set_size, train_size, *args, **kwargs):
        self.reference_set_size = reference_set_size
        kwargs['module__num_M'] = train_size-reference_set_size
        super().__init__(criterion=IdentityLoss, *args, **kwargs)

    def predict(self, X):
        self.module_.eval()

        samples = 30
        dxy = np.zeros((X.shape[0], samples))

        with torch.no_grad():
            for j in range(samples):
                dxy[:, j] = to_numpy(self.module_.predict(to_tensor(X, device=self.device), self.xR, self.yR).squeeze(1))

        mean_dxy, std_dxy = dxy.mean(axis=1), dxy.std(axis=1)

        return mean_dxy, std_dxy

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
        idx = np.arange(X.shape[0])
        idx_r = np.random.choice(idx, size=(self.reference_set_size,), replace=False)
        idx_m = np.array([i for i in np.arange(X.shape[0]) if i not in idx_r])
        self.xR, self.yR = to_tensor(X[idx_r], device=self.device), to_tensor(y[idx_r], device=self.device)
        x_m, y_m = X[idx_m], y[idx_m]
        m_dict = {'XM': x_m, 'yM': y_m}
        
        super(RegressionFNPSkorch, self).fit(m_dict, y_m, **fit_params)


# simple Helper loss because the fnp implementation already returns built in loss on forward pass
class IdentityLoss(nn.Module):
    def __init__(self):
        super(IdentityLoss, self).__init__()

    def forward(self, preds, target):
        return preds