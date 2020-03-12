import torch
import torch.nn as nn

import torchbnn as bnn

from models.base_nn import BaseNN

'''
Simple boilerplate Bayesian Neural Network implementation using torchbnn.
Supports supplying with dynamic amount of layer counts and sizes 
'''


class TorchBNN(BaseNN):
    def __init__(self, input_size, output_size, prior_mu, prior_sigma, **kwargs):
        super(TorchBNN, self).__init__(**kwargs)

        self.hidden_size.insert(0, input_size)

        self.layers = nn.ModuleList([
            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma,
                            in_features=self.hidden_size[i], out_features=self.hidden_size[i + 1])
            for i in range(len(self.hidden_size) - 1)])

        self.last_layer = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=self.hidden_size[-1],
                                          out_features=output_size)

    def forward(self, x, dropout_rate=0):
        out = x
        for l in self.layers:
            out = l(out)
            out = torch.relu(out)

        out = self.last_layer(out)

        return out
