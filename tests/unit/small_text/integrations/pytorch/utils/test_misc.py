import unittest
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch
    from small_text.integrations.pytorch.utils.misc import default_tensor_type
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class MiscUtilsTest(unittest.TestCase):

    def test_default_tensor_type(self):
        with self.assertWarnsRegex(DeprecationWarning, r'default_tensor_type\(\) is deprecated'):
            with default_tensor_type(torch.FloatTensor):
                torch.Tensor([0])
