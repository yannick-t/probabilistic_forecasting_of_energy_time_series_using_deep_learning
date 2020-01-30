from skorch import NeuralNet
import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor

from models.skorch_wrappers.aleotoric_nn_skorch import AleotoricNNSkorch


# wrapper for concrete dropout to supply lengthscale and automatically calculate the regularization parameters
class ConcreteSkorch(AleotoricNNSkorch):
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



