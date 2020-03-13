import torch
import torch.nn as nn

from models.base_nn import BaseNN

'''
single nn to use as one of many in Deep Ensemble
Implementation of concepts from Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
Simple and scalable predictive uncertainty estimation using deep ensembles.
In Advances in neural information processing systems (pp. 6402-6413).
'''


class DeepEnsembleSingle(BaseNN):
    def __init__(self, input_size, output_size, **kwargs):
        super(DeepEnsembleSingle, self).__init__(**kwargs)

        self.hidden_size.insert(0, input_size)

        self.layers = nn.ModuleList([nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]) for i in range(len(self.hidden_size) - 1)])
        self.last_layer = nn.Linear(self.hidden_size[-1], output_size)

        for l in self.layers:
            self.init_layer(l)
        self.init_layer(self.last_layer)

    def init_layer(self, layer):
        # initialize parameters randomly according to paper
        nn.init.xavier_uniform_(layer.weight)
        nn.init.uniform_(layer.bias, -0.2, 0.2)

    def forward(self, x):
        out = x
        for l in self.layers:
            out = l(out)
            out = torch.relu(out)

        out = self.last_layer(out)
        return out

