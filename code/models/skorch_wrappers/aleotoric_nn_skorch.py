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
        dxy = torch.zeros((X.shape[0], self.sample_count, output_size))

        with torch.no_grad():
            for j in range(self.sample_count):
                dxy[:, j] = to_tensor(self.predict_proba(to_tensor(X, device=self.device)), self.device)

        # assuming network predicts mean and std of aleotoric uncertainty
        # first mean of all output dims then std of all output dims
        # to be used with Heteroscedastic loss class
        # combined mean
        outut_dim = int(output_size / 2)
        mean = dxy[..., :outut_dim].mean(dim=1)

        # use sigmoid to be numerically stable and not depend on activation functions of nn
        aleo_sampled = torch.sigmoid(dxy[..., outut_dim:])

        # combined approx variance from paper
        # Kendall, A., & Gal, Y. (2017).
        # What uncertainties do we need in bayesian deep learning for computer vision?.
        # In Advances in neural information processing systems (pp. 5574-5584).
        var = (dxy[..., :outut_dim]**2).mean(dim=1) - mean**2 + (aleo_sampled**2).mean(dim=1)

        epistemic_std = dxy[..., :outut_dim].std(dim=1)
        aleotoric_std = aleo_sampled.mean(dim=1)

        if torch.isnan(mean).any():
            print(mean)

        return np.stack([to_numpy(mean), to_numpy(var), to_numpy(epistemic_std), to_numpy(aleotoric_std)], -1)
