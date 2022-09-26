import unittest
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from torch.nn import BCEWithLogitsLoss

    from small_text.integrations.pytorch.utils.loss import _LossAdapter2DTo1D
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class LossTest(unittest.TestCase):

    def test_loss_fct(self):
        loss_fct = _LossAdapter2DTo1D(BCEWithLogitsLoss(reduction='none'))

        input = torch.randn(5, 3)
        target = torch.empty(5, 3).random_(2)

        loss = loss_fct(input, target)

        self.assertEqual(1, len(loss.shape))
        self.assertEqual(5, loss.shape[0])
