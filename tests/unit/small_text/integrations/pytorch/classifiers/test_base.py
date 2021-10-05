import pytest
import unittest

import numpy as np

from unittest.mock import patch
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import random_text_classification_dataset

try:
    import torch

    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from small_text.integrations.pytorch.classifiers import PytorchClassifier
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier

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


class _PytorchClassifierBaseFunctionalityTest(object):

    def _get_clf(self):
        raise NotImplementedError()

    def test_predict_on_empty_data(self):
        train_set = random_text_classification_dataset(10)
        test_set = PytorchTextClassificationDataset([], None)

        clf = self._get_clf()
        clf.fit(train_set)

        predictions = clf.predict(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        train_set = random_text_classification_dataset(10)
        test_set = PytorchTextClassificationDataset([], None)

        clf = self._get_clf()
        clf.fit(train_set)

        predictions, proba = clf.predict_proba(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))
        self.assertEqual(0, proba.shape[0])
        self.assertTrue(np.issubdtype(proba.dtype, np.float))


@pytest.mark.pytorch
class KimCNNBaseFunctionalityTest(unittest.TestCase, _PytorchClassifierBaseFunctionalityTest):

    def _get_clf(self):
        embedding_matrix = torch.rand(5, 60)
        return KimCNNClassifier(embedding_matrix=embedding_matrix, device='cpu')


@pytest.mark.pytorch
class SimplePytorchClassifierTest(unittest.TestCase):

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
