import torch.nn as nn


class BaseNN(nn.Module):
    # simple utilty abstract class to be able to pass hidden layer sizes as separate arguments
    # as well as array
    def __init__(self, **kwargs):
        super(BaseNN, self).__init__()

        if 'hidden_size' not in kwargs:
            hidden_size = []
            for i in range(0, 6):
                key = 'hidden_size_%d' % i
                if key in kwargs and kwargs[key] != 0:
                    hidden_size.append(kwargs[key])
        else:
            hidden_size = kwargs['hidden_size']

        self.hidden_size = hidden_size

