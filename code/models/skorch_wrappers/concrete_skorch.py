from skorch import NeuralNet
import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor

from models.skorch_wrappers.aleotoric_nn_skorch import AleotoricNNSkorch


# wrapper for concrete dropout to supply lengthscale and automatically calculate the regularization parameters
class ConcreteSkorch(AleotoricNNSkorch):
    def __init__(self, lengthscale, dataset_size, *args, **kwargs):
        # calculate regularization parameters according to paper
        if 'module__weight_regularizer' not in kwargs:
            wr = lengthscale ** 2. / dataset_size
            dr = 2. / dataset_size

            kwargs['module__weight_regularizer'] = wr
            kwargs['module__dropout_regularizer'] = dr

        self.lengthscale = lengthscale
        self.dataset_size = dataset_size

        super().__init__(*args, **kwargs)



