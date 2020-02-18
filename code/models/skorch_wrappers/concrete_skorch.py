from skorch import NeuralNet
import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor

from models.skorch_wrappers.combined_unc_nn_skorch import CombinedUncNNSkorch


# wrapper for concrete dropout to supply lengthscale and automatically calculate the regularization parameters
class ConcreteSkorch(CombinedUncNNSkorch):
    def __init__(self, lengthscale, dataset_size, *args, **kwargs):

        self.lengthscale = lengthscale
        self.dataset_size = dataset_size

        super().__init__(*args, **kwargs)

    def calc_params(self):
        # calculate regularization parameters according to paper

        wr = self.lengthscale ** 2. / self.dataset_size
        dr = 2. / self.dataset_size

        self.module__weight_regularizer = wr
        self.module__dropout_regularizer = dr
        
    def initialize_module(self):
        self.calc_params()
        super(ConcreteSkorch, self).initialize_module()

    def get_loss(self, y_pred, y_true, X=None, training=False):
        # add regularization that concrete dropout returns to loss
        return super(ConcreteSkorch, self).get_loss(y_pred[0], y_true, X, training) + y_pred[1]



