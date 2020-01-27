# code adapted from https://github.com/cornellius-gp/gpytorch/blob/master/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.ipynb
#
# License:
# MIT License
#
# Copyright (c) 2017 Jake Gardner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import gpytorch
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood, DeepGPLayer, DeepGP
import torch
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch import nn
from torch.nn import Linear


class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

        self.linear_layer = Linear(input_dims, 1)

    def forward(self, x):
        mean_x = self.mean_module(x) # self.linear_layer(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGaussianProcess(DeepGP):
    def __init__(self, input_size, output_size, num_inducing=128, **kwargs):
        # pass hidden layer sizes as separate arguments as well as array
        if 'hidden_size' not in kwargs:
            hidden_size = []
            for i in range(0, 6):
                key = 'hidden_size_%d' % i
                if key in kwargs and kwargs[key] != 0:
                    hidden_size.append(kwargs[key])
        else:
            hidden_size = kwargs['hidden_size']

        self.hidden_size = hidden_size
        self.hidden_size.append(output_size)

        first_layer = DeepGPHiddenLayer(
                    input_dims=input_size,
                    output_dims=self.hidden_size[0],
                    mean_type='linear',
                    num_inducing=num_inducing
                )

        # variable count of hidden layers and neurons
        hidden_layers = nn.ModuleList(
            [
                DeepGPHiddenLayer(
                    input_dims=self.hidden_size[i],
                    output_dims=self.hidden_size[i + 1],
                    mean_type='constant',
                    num_inducing=num_inducing
                )
                for i in range(len(self.hidden_size) - 1)
            ])

        super().__init__()

        self.first_layer = first_layer
        self.hidden_layers = hidden_layers

    def forward(self, inputs):
        out = self.first_layer(inputs)
        for hidden in self.hidden_layers:
            out = hidden(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            mus = []
            variances = []
            preds = self(x)
            mus.append(preds.mean)
            variances.append(preds.variance)

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1)
