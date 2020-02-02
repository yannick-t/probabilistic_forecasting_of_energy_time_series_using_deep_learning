import torch
import torch.nn as nn

from models.base_nn import BaseNN


class SimpleNN(BaseNN):
    def __init__(self, input_size, output_size, **kwargs):
        super(SimpleNN, self).__init__(**kwargs)

        self.fc1 = nn.Linear(input_size, self.hidden_size[0])
        self.fcs = nn.ModuleList([nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]) for i in range(len(self.hidden_size) - 1)])
        self.fc2 = nn.Linear(self.hidden_size[-1], output_size)

        nn.init.xavier_uniform_(self.fc1.weight)
        for fc in self.fcs:
            nn.init.xavier_uniform_(fc.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x, dropout_rate=0):
        out = self.fc1(x)
        out = torch.sigmoid(out)

        for fc in self.fcs:
            out = fc(out)
            out = torch.sigmoid(out)

        out = torch.dropout(out, dropout_rate, True)

        out = self.fc2(out)
        return out

