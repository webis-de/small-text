import unittest
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    import torch.nn as nn
    from small_text.integrations.pytorch.utils.misc import enable_dropout


    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()

            self.dropout1 = nn.Dropout()
            self.dropout2 = nn.Dropout2d()
            self.fc = nn.Linear(10, 4)

except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class MiscUtilsTest(unittest.TestCase):

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
