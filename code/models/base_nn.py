import torch.nn as nn


class BaseNN(nn.Module):
    # simple utilty abstract class to be able to pass hidden layer sizes as separate arguments
    # as well as array
    def __init__(self, **kwargs):
        super(BaseNN, self).__init__()

        hidden_size = hidden_size_extract(kwargs, 'hidden_size')

        self.hidden_size = hidden_size


def hidden_size_extract(kwargs, name, delete_from_dict=False):
    if name not in kwargs:
        hidden_size = []
        for i in range(0, 6):
            key = name + '_%d' % i
            if key in kwargs and kwargs[key] != 0:
                hidden_size.append(kwargs[key])

                if delete_from_dict:
                    kwargs.pop(key)
    else:
        hidden_size = kwargs[name].copy()

        if delete_from_dict:
            kwargs.pop(name)

    return hidden_size