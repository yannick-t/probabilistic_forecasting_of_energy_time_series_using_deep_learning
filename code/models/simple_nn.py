import torch
import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None, **kwargs):
        super(SimpleNN, self).__init__()

        if hidden_size is None:
            hidden_size = []
            for i in range(0, 6):
                key = 'hidden_size_%d' % i
                if key in kwargs and kwargs[key] != 0:
                    hidden_size.append(kwargs[key])

        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.fcs = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i + 1]) for i in range(len(hidden_size) - 1)])
        self.fc2 = nn.Linear(hidden_size[-1], output_size)

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
        out = torch.sigmoid(out)
        return out

