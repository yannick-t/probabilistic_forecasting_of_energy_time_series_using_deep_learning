from training.loss.heteroscedastic_loss import HeteroscedasticLoss


class ConcreteHeteroscedasticLoss(HeteroscedasticLoss):
    def __init__(self):
        super().__init__()

    def forward(self, preds, target):
        # add regularization that concrete dropout returns to loss
        preds, regularization = preds

        return super(ConcreteHeteroscedasticLoss, self).forward(preds, target) + regularization
