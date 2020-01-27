# Code based on https://github.com/yaringal/ConcreteDropout
# License:
# MIT License
#
# Copyright (c) 2017
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

import torch
import numpy as np
from torch import nn

from models.base_nn import BaseNN


class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)

        out = layer(self._concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x


class ConcreteDropoutNN(BaseNN):
    def __init__(self, weight_regularizer, dropout_regularizer, input_size, output_size, **kwargs):
        super(ConcreteDropoutNN, self).__init__(**kwargs)

        self.hidden_size.append(output_size)

        self.linear1 = nn.Linear(input_size, self.hidden_size[0])
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]) for i in range(len(self.hidden_size) - 1)])

        self.conc_drops = nn.ModuleList([ConcreteDropout(weight_regularizer=weight_regularizer,
                                                         dropout_regularizer=dropout_regularizer)
                                         for i in range(len(self.hidden_size))])

        self.act = nn.ReLU()

        nn.init.xavier_uniform_(self.linear1.weight)
        for fc in self.linears:
            nn.init.xavier_uniform_(fc.weight)

    def forward(self, x):
        regularization = torch.empty(len(self.hidden_size), device=x.device)

        out, regularization[0] = self.conc_drops[0](x, nn.Sequential(self.linear1, self.act))
        for i in range(len(self.hidden_size) - 1):
            if i == len(self.hidden_size) - 2:
                act = nn.Sigmoid()
            else:
                act = self.act
            out, regularization[i + 1] = self.conc_drops[i + 1](out, nn.Sequential(self.linears[i], act))

        return out, regularization.sum()
