import numpy as np
import torch
from skorch.utils import to_numpy, to_tensor

from models.skorch_wrappers.base_nn_skorch import BaseNNSkorch


# wrapper for simple aleatoric uncertainty nn
# trained with Heteroscedastic or CRPS Loss
class AleatoricNNSkorch(BaseNNSkorch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, X):
        self.module_.eval()

        output_size = self.module__output_size
        assert output_size % 2 == 0
        outut_dim = int(output_size / 2)

        pred = to_tensor(self.predict_proba(to_tensor(X, device=self.device)), self.device)

        mean = pred[..., :outut_dim]
        # use softplus to be numerically stable and not depend on activation functions of nn
        softplus = torch.nn.Softplus()
        std = softplus(pred[..., outut_dim:])

        return np.stack([to_numpy(mean), to_numpy(std**2)], -1)
