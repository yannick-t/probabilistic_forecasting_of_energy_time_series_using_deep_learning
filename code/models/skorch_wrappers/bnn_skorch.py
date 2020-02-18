from skorch.utils import to_tensor
from torchbnn import BKLLoss
from models.skorch_wrappers.combined_unc_nn_skorch import CombinedUncNNSkorch


# wrapper for bnn (mainly because of loss)
class BNNSkorch(CombinedUncNNSkorch):
    def __init__(self, bkl=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Variatonal inference (sums up kl loss for layers)
        if bkl is not None:
            self.bkl = bkl
        else:
            self.bkl = BKLLoss()

    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        return self.criterion_(y_pred, y_true) + self.bkl(self.module_)

