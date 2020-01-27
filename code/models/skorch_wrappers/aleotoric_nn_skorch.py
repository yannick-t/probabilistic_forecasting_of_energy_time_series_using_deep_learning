from skorch import NeuralNet
import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor

from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch


# wrapper to combine aleotoric and  epistemic uncertainties in one for most models
# using sampling
class AleotoricNNSkorch(BaseNNSkorch):
    def __init__(self, sample_count, *args, **kwargs):
        self.sample_count = sample_count
        super().__init__(*args, **kwargs)

    def predict(self, X):
        self.module_.eval()

        output_size = self.module__output_size
        assert output_size % 2 == 0
        dxy = np.zeros((X.shape[0], self.sample_count, output_size))

        with torch.no_grad():
            for j in range(self.sample_count):
                dxy[:, j] = to_numpy(self.predict_proba(to_tensor(X, device=self.device)))

        # assuming network predicts mean and std of aleotoric uncertainty
        # first mean of all output dims then std of all output dims
        # to be used with Heteroscedastic loss class
        # combined mean
        outut_dim = int(output_size / 2)
        mean = dxy[..., :outut_dim].mean(axis=1)
        # combined approx variance from paper
        # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
        var = (dxy[..., :outut_dim]**2).mean(axis=1) - mean**2 + (dxy[..., outut_dim:]**2).mean(axis=1)

        epistemic_std = dxy[..., :outut_dim].std(axis=1)
        aleotoric_std = np.mean(dxy[..., outut_dim:], 1)

        return np.stack([mean, var, epistemic_std, aleotoric_std], -1)
