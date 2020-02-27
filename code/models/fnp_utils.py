# Functional Neural Process code adapted from https://github.com/AMLab-Amsterdam/FNP
# by AMLAB Amsterdam
#
# License:
#
# MIT License
#
# Copyright (c) 2019 AMLAB
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
import math
import torch.nn.functional as F
import numpy as np
from torch.distributions import Bernoulli
from itertools import product


def logitexp(logp):
    # https: // github.com / pytorch / pytorch / issues / 4007
    pos = torch.clamp(logp, min=-0.69314718056)
    neg = torch.clamp(logp, max=-0.69314718056)
    neg_val = neg - torch.log(1 - torch.exp(neg))
    pos_val = -torch.log(torch.clamp(torch.expm1(-pos), min=1e-20))
    return pos_val + neg_val


def one_hot(x, n_classes=10):
    x_onehot = torch.Tensor(x.size(0), n_classes, device=x.device).zero_()
    x_onehot.scatter_(1, x[:, None], 1)

    return x_onehot


class LogitRelaxedBernoulli(object):
    def __init__(self, logits, temperature=0.3, **kwargs):
        self.logits = logits
        self.temperature = temperature

    def rsample(self):
        eps = torch.clamp(torch.rand(self.logits.size(), dtype=self.logits.dtype, device=self.logits.device),
                          min=1e-6, max=1-1e-6)
        y = (self.logits + torch.log(eps) - torch.log(1. - eps)) / self.temperature
        return y

    def log_prob(self, value):
        return math.log(self.temperature) - self.temperature * value + self.logits \
               - 2 * F.softplus(-self.temperature * value + self.logits)


class Normal(object):
    def __init__(self, means, logscales, **kwargs):
        self.means = means
        self.logscales = logscales

    def log_prob(self, value):
        log_prob = torch.pow(value - self.means, 2)
        log_prob *= - (1 / (2. * self.logscales.mul(2.).exp()))
        log_prob -= self.logscales + .5 * math.log(2. * math.pi)
        return log_prob

    def sample(self, **kwargs):
        eps = torch.normal(torch.Tensor(self.means.size()).zero_(), torch.Tensor(self.means.size()).fill_(1))
        eps = eps.to(self.means.device)
        return self.means + self.logscales.exp() * eps

    def rsample(self, **kwargs):
        return self.sample(**kwargs)


def order_z(z):
    # scalar ordering function
    if z.size(1) == 1:
        return z
    log_cdf = torch.sum(torch.log(.5 + .5 * torch.erf(z / math.sqrt(2))), dim=1, keepdim=True)
    return log_cdf


def sample_DAG(Z, g, training=True, temperature=0.3):
    # get the indices of an upper triangular adjacency matrix that represents the DAG
    idx_utr = np.triu_indices(Z.size(0), 1)

    # get the ordering
    ordering = order_z(Z)
    # sort the latents according to the ordering
    sort_idx = torch.sort(torch.squeeze(ordering), 0)[1]
    Y = Z[sort_idx, :]
    # form the latent pairs for the edges
    Z_pairs = torch.cat([Y[idx_utr[0]], Y[idx_utr[1]]], 1)
    # get the logits for the edges in the DAG
    logits = g(Z_pairs)

    if training:
        p_edges = LogitRelaxedBernoulli(logits=logits, temperature=temperature)
        G = torch.sigmoid(p_edges.rsample())
    else:
        p_edges = Bernoulli(logits=logits)
        G = p_edges.sample()

    # embed the upper triangular to the adjacency matrix
    unsorted_G = torch.Tensor(Z.size(0), Z.size(0)).zero_().to(Z.device)
    unsorted_G[idx_utr[0], idx_utr[1]] = G.squeeze()
    # unsort the dag to conform to the data order
    original_idx = torch.sort(sort_idx)[1]
    unsorted_G = unsorted_G[original_idx, :][:, original_idx]

    return unsorted_G


def sample_bipartite(Z1, Z2, g, training=True, temperature=0.3):
    indices = []
    for element in product(range(Z1.size(0)), range(Z2.size(0))):
        indices.append(element)
    indices = np.array(indices)
    Z_pairs = torch.cat([Z1[indices[:, 0]], Z2[indices[:, 1]]], 1)

    logits = g(Z_pairs)
    if training:
        p_edges = LogitRelaxedBernoulli(logits=logits, temperature=temperature)
        A_vals = torch.sigmoid(p_edges.rsample())
    else:
        # workaround for hyperparameter optimization to keep going if fnp produces
        # nan here for some reason
        if torch.isnan(logits).any():
            return None
        p_edges = Bernoulli(logits=logits, validate_args=True)
        A_vals = p_edges.sample()

    # embed the values to the adjacency matrix
    A = torch.Tensor(Z1.size(0), Z2.size(0)).zero_().to(Z1.device)
    A[indices[:, 0], indices[:, 1]] = A_vals.squeeze()

    return A


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        assert len(x.shape) > 1

        return x.view(x.shape[0], -1)

