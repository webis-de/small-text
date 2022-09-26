from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from torch.nn.modules.loss import _Loss
except ImportError:
    raise PytorchNotFoundError('Could not import pytorch')


class _LossAdapter2DTo1D(_Loss):

    def __init__(self, base_loss_fct):
        super().__init__()
        self.base_loss_fct = base_loss_fct
        self.reduction = self.base_loss_fct.reduction

    def forward(self, input, target):
        return self.base_loss_fct(input, target).mean(dim=-1)
