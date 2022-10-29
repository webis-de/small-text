import unittest
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    import torch.nn as nn
    from small_text.integrations.pytorch.utils.misc import default_tensor_type, enable_dropout
except (ImportError, PytorchNotFoundError):
    pass


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout2d()
        self.fc = nn.Linear(10, 4)


@pytest.mark.pytorch
class MiscUtilsTest(unittest.TestCase):

    def test_default_tensor_type(self):
        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The function default_tensor_type has been deprecated in 1.1.0 '
                                   r'and will be removed in 2.0.0.'):
            with default_tensor_type(torch.FloatTensor):
                torch.Tensor([0])

    def test_enable_dropout(self):
        model = SimpleNet()

        model.eval()
        self.assertFalse(model.dropout1.training)
        self.assertFalse(model.dropout2.training)

        model.dropout2.train()
        self.assertTrue(model.dropout2.training)

        with enable_dropout(model):
            self.assertTrue(model.dropout1.training)
            self.assertTrue(model.dropout2.training)

        self.assertFalse(model.dropout1.training)
        self.assertTrue(model.dropout2.training)
