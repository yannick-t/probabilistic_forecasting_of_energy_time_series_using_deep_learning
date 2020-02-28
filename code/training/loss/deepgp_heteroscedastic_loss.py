import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from torch import nn
from torch.distributions import Normal


class DeepGPHeteroscedasticLoss(nn.Module):
    def __init__(self, likelihood, model, num_data, base_loss):
        super().__init__()
        self.mll = DeepApproximateMLL(VariationalELBO(likelihood=likelihood, model=model, num_data=num_data))
        self.base_loss = base_loss

    def forward(self, preds, target):
        assert not target.requires_grad

        # combine heteroscedastic loss with deep gp variational inference
        # first mean of all output dims then var of all output dims

        outut_dim = int(preds.mean.shape[-1] / 2)
        mean = preds.mean[..., :outut_dim]
        std = preds.mean[..., outut_dim:]

        std = torch.sigmoid(std)

        # base loss
        loss = -self.base_loss(mean, std, target)
        loss = loss.mean(0)
        loss = loss.mean(-1)
        covar_shape = preds.covariance_matrix.shape

        pred_aleo_mean = MultivariateNormal(preds.mean[..., :outut_dim].reshape([-1, int(covar_shape[-1] / 2)]),
                                            preds.covariance_matrix[..., :int(covar_shape[-2] / 2), :int(covar_shape[-1] / 2)])

        return self.mll(pred_aleo_mean, target.reshape([-1, pred_aleo_mean.loc.shape[-1]])) + loss
