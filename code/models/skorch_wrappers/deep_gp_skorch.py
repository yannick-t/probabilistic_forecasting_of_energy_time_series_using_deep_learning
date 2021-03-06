from gpytorch.likelihoods import GaussianLikelihood
from skorch import NeuralNet
import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor

from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch
from training.loss.deepgp_heteroscedastic_loss import DeepGPHeteroscedasticLoss
from training.loss.torch_loss_fns import bnll


class DeepGPSkorch(BaseNNSkorch):
    def __init__(self, num_data, base_loss=bnll, *args, **kwargs):
        self.num_data = num_data
        self.base_loss = base_loss
        kwargs['criterion'] = None
        self.module__likelihood = GaussianLikelihood()
        super().__init__(*args, **kwargs)

    def initialize(self):
        """Initializes all components of the :class:`.NeuralNet` and
        returns self.

        """
        self.initialize_virtual_params()
        self.initialize_callbacks()
        # different order to init criterion with model
        self.initialize_module()
        self.initialize_criterion()
        self.initialize_optimizer()
        self.initialize_history()

        self.initialized_ = True
        return self

    def initialize_criterion(self):
        """Initializes the criterion."""
        criterion_params = self._get_params_for('criterion')
        self.criterion_ = DeepGPHeteroscedasticLoss(likelihood=self.module__likelihood, model=self.module_, num_data=self.num_data, base_loss=self.base_loss)
        if isinstance(self.criterion_, torch.nn.Module):
            self.criterion_ = self.criterion_.to(self.device)
        return self

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        return -self.criterion_(y_pred, y_true)

    def predict(self, X):
        self.module_.eval()

        # batched predictions, because ram!
        pred_means = []
        pred_vars = []
        for idx in np.arange(0, X.shape[0], self.batch_size):
            predictive_means, predictive_variances = \
                self.module_.predict(to_tensor(X[idx: min(idx + self.batch_size, X.shape[0])], self.device))

            pred_means.append(predictive_means)
            pred_vars.append(predictive_variances)

        predictive_means = torch.cat(pred_means, dim=1)
        predictive_variances = torch.cat(pred_vars, dim=1)

        outut_dim = int(self.module__output_size / 2)

        mean = predictive_means.mean(0)[..., :outut_dim]
        epistemic_var = predictive_variances.mean(0)[..., :outut_dim]
        softplus = torch.nn.Softplus()
        aleotoric_var = softplus(predictive_means.mean(0)[..., outut_dim:])**2
        var = epistemic_var + aleotoric_var

        return np.stack(
            [to_numpy(mean),
             to_numpy(var),
             to_numpy(epistemic_var),
             to_numpy(aleotoric_var)],
            -1)


