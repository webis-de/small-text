import unittest

import pytest

from unittest.mock import patch
from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError


try:
    from active_learning.integrations.pytorch.classifiers import PytorchClassifier

    class SimplePytorchClassifier(PytorchClassifier):
        """Simple subclass to allow instantiation."""

        def fit(self, train_set, _=None, *args, **kwargs):
            pass

        def validate(self, validation_set):
            pass

        def predict(self, test_set, return_proba=False):
            pass

        def predict_proba(self, test_set):
            pass
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class PytorchClassifierTest(unittest.TestCase):

    @patch('torch.cuda.is_available')
    def test_default_init(self, mock_is_available):
        mock_is_available.return_value = False

        clf = SimplePytorchClassifier()
        self.assertEqual('cpu', clf.device)
        mock_is_available.assert_called_with()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_init_with_cuda_available(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(device='cuda')
        self.assertEqual('cuda', clf.device)
        mock_is_available.assert_called_with()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_init_with_cuda_available(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier()
        self.assertEqual('cuda', clf.device)
        mock_is_available.assert_called_with()
