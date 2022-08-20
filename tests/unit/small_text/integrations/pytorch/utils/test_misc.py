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
        with self.assertWarnsRegex(DeprecationWarning,
                                   r'The function default_tensor_type has been deprecated in 1.1.0 '
                                   r'and will be removed in 2.0.0.'):
            with default_tensor_type(torch.FloatTensor):
                torch.Tensor([0])
