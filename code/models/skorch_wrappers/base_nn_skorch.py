from skorch import NeuralNet
import torch

# base skorch wrapper to automatically set default tensor type / device
# for prallel processing, because the defaults need to be set on each process


class BaseNNSkorch(NeuralNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        torch.set_default_tensor_type('torch.' + ('cuda.' if self.device == 'cuda' else '') + 'DoubleTensor')

    def set_params(self, **kwargs):
        super(BaseNNSkorch, self).set_params(**kwargs)
        torch.set_default_tensor_type('torch.' + ('cuda.' if self.device == 'cuda' else '') + 'DoubleTensor')

    def get_iterator(self, dataset, training=False):
        if training:
            kwargs = self._get_params_for('iterator_train')
            iterator = self.iterator_train
        else:
            kwargs = self._get_params_for('iterator_valid')
            iterator = self.iterator_valid

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size

        if kwargs['batch_size'] == -1:
            kwargs['batch_size'] = len(dataset)

        # Basic fix to make skopt compatible with torch because skopt generates np ints and torch dataloaders
        # don't accept them for batch size
        if 'batch_size' in kwargs:
            kwargs['batch_size'] = int(kwargs['batch_size'])

        return iterator(dataset, **kwargs)

    def fit(self, X, y=None, **fit_params):
        torch.set_default_tensor_type('torch.' + ('cuda.' if self.device == 'cuda' else '') + 'DoubleTensor')
        super(BaseNNSkorch, self).fit(X, y, **fit_params)
