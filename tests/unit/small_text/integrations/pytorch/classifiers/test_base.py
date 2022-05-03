import pytest
import unittest

from unittest.mock import patch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import random_text_classification_dataset

try:
    import torch
    from torch.nn import CrossEntropyLoss
    from torch.nn import BCEWithLogitsLoss

    from small_text.integrations.pytorch.classifiers import PytorchClassifier

    class SimplePytorchClassifier(PytorchClassifier):
        """Simple subclass to allow instantiation."""

        def __init__(self, num_classes, multi_label=False, class_weight=None, device=None):
            self.num_classes = num_classes
            self.class_weight = class_weight
            super().__init__(multi_label=multi_label, device=device)

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
class SimplePytorchClassifierTest(unittest.TestCase):

    @patch('torch.cuda.is_available')
    def test_default_init(self, mock_is_available):
        mock_is_available.return_value = False

        clf = SimplePytorchClassifier(2)
        self.assertFalse(clf.multi_label)
        self.assertEqual('cpu', clf.device)
        mock_is_available.assert_called_with()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_init_with_cuda_available_and_device(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(2, device='cuda')
        self.assertFalse(clf.multi_label)
        self.assertEqual('cuda', clf.device)
        mock_is_available.assert_called_with()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_init_with_cuda_available(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(2)
        self.assertFalse(clf.multi_label)
        self.assertEqual('cuda', clf.device)
        mock_is_available.assert_called_with()

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_binary(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(2)
        clf.class_weights_ = torch.ones(2)
        loss = clf.get_default_criterion()
        self.assertTrue(isinstance(loss, BCEWithLogitsLoss))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_multiclass(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3)
        clf.class_weights_ = torch.ones(3)
        loss = clf.get_default_criterion()
        self.assertTrue(isinstance(loss, CrossEntropyLoss))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_multilabel(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3, multi_label=True)
        clf.class_weights_ = torch.ones(3)
        loss = clf.get_default_criterion()
        self.assertTrue(isinstance(loss, BCEWithLogitsLoss))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_initialize_class_weights(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3)
        train_set = random_text_classification_dataset(max_length=10, num_classes=3)
        class_weights = clf.initialize_class_weights(train_set)

        self.assertIsNone(class_weights)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_initialize_class_weights_balanced(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3, multi_label=True, class_weight='balanced')
        train_set = random_text_classification_dataset(max_length=10, num_classes=3)
        class_weights = clf.initialize_class_weights(train_set)

        self.assertIsNotNone(class_weights)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_initialize_class_weights_invalid_value(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3, class_weight='does_not_exist')
        train_set = random_text_classification_dataset(max_length=10, num_classes=3)

        with self.assertRaisesRegex(ValueError, 'Invalid value for class_weight'):
            clf.initialize_class_weights(train_set)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_initialize_class_weights_multi_label_warning(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3, multi_label=True, class_weight='balanced')
        train_set = random_text_classification_dataset(max_length=10, num_classes=3)

        with self.assertWarnsRegex(UserWarning, 'Setting class_weight to \'balanced\' is intended'):
            clf.initialize_class_weights(train_set)

    def test_sum_up_accuracy(self):
        logits = torch.FloatTensor([
            [2.22, -0.14, 0.13],
            [0.12, 1.05, 3.13],
            [-0.56, 0.19, 1.02]
        ], device='cpu')

        cls = torch.IntTensor([0, 2, 1])
        clf = SimplePytorchClassifier(3)

        accuracy = clf.sum_up_accuracy_(logits, cls)
        self.assertEqual(2, accuracy)

    def test_sum_up_accuracy_multi_label(self):
        logits = torch.FloatTensor([
            [2.22, -0.14, 0.13],
            [0.12, 1.05, 3.13],
            [-0.56, 0.19, 1.02]
        ], device='cpu')

        cls = torch.IntTensor([
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
        ])
        clf = SimplePytorchClassifier(3, multi_label=True)

        accuracy = clf.sum_up_accuracy_(logits, cls)
        self.assertAlmostEqual(2.333, accuracy, places=3)
