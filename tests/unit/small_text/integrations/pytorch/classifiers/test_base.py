import pytest
import unittest

from unittest.mock import Mock, patch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.training.model_selection import ModelSelection, NoopModelSelection
from tests.utils.datasets import random_text_classification_dataset

try:
    import torch
    from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

    from small_text.integrations.pytorch.classifiers import PytorchClassifier
    from small_text.integrations.pytorch.utils.loss import _LossAdapter2DTo1D

    class SimplePytorchClassifier(PytorchClassifier):
        """Simple subclass to allow instantiation."""

        def __init__(self, num_classes, multi_label=False, class_weight=None, device=None):
            self.num_classes = num_classes
            self.class_weight = class_weight
            super().__init__(multi_label=multi_label, device=device)

        def fit(self, train_set, _=None, *args, **kwargs):
            pass


except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class PytorchClassifierTest(unittest.TestCase):

    def test_predict_proba(self):
        class PytorchClassifierImpl(PytorchClassifier):

            def __init__(self):
                self.model = None
                self._create_collate_fn = None
                super().__init__()

            def fit(self, train_set, _=None, *args, **kwargs):
                self.model = Mock()

                def create_collate_fn():
                    def collate(dataset):
                        return torch.arange(0, len(dataset))
                    return collate
                self._create_collate_fn = create_collate_fn

        clf = PytorchClassifierImpl()
        ds = random_text_classification_dataset()

        clf.fit(ds)

        with self.assertRaisesRegex(NotImplementedError, r'_predict_proba\(\) needs to be implemented'):
            clf.predict_proba(ds)

    def test_predict_proba_dropout(self):
        class PytorchClassifierImpl(PytorchClassifier):

            def __init__(self):
                self.model = None
                self._create_collate_fn = None
                super().__init__()

            def fit(self, train_set, _=None, *args, **kwargs):
                self.model = Mock()

                def create_collate_fn():
                    def collate(dataset):
                        return torch.arange(0, len(dataset))
                    return collate

                self._create_collate_fn = create_collate_fn

        clf = PytorchClassifierImpl()
        ds = random_text_classification_dataset()

        clf.fit(ds)

        with self.assertRaisesRegex(NotImplementedError,
                                    r'_predict_proba_dropout_sampling\(\) needs to be implemented'):
            clf.predict_proba(ds, dropout_sampling=3)


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
        class_weights = torch.ones(2)
        loss = clf._get_default_criterion(class_weights)
        self.assertTrue(isinstance(loss, BCEWithLogitsLoss))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_multiclass(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3)
        class_weights = torch.ones(3)
        loss = clf._get_default_criterion(class_weights)
        self.assertTrue(isinstance(loss, CrossEntropyLoss))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_multilabel(self, mock_current_device, mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3, multi_label=True)
        class_weights = torch.ones(3)
        loss = clf._get_default_criterion(class_weights)
        self.assertTrue(isinstance(loss, BCEWithLogitsLoss))

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_use_sample_weights_binary(self,
                                                             mock_current_device,
                                                             mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(2)
        class_weights = torch.ones(2)
        loss = clf._get_default_criterion(class_weights, use_sample_weights=True)
        self.assertTrue(isinstance(loss, _LossAdapter2DTo1D))
        self.assertEqual('none', loss.reduction)

        self.assertTrue(isinstance(loss.base_loss_fct, BCEWithLogitsLoss))
        self.assertEqual('none', loss.base_loss_fct.reduction)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_use_sample_weights_multiclass(self,
                                                                 mock_current_device,
                                                                 mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3)
        class_weights = torch.ones(3)
        loss = clf._get_default_criterion(class_weights, use_sample_weights=True)
        self.assertTrue(isinstance(loss, CrossEntropyLoss))
        self.assertEqual('none', loss.reduction)

    @patch('torch.cuda.is_available')
    @patch('torch.cuda.current_device')
    def test_get_default_criterion_use_sample_weights_multilabel(self,
                                                                 mock_current_device,
                                                                 mock_is_available):
        mock_current_device.return_value = '0'
        mock_is_available.return_value = True

        clf = SimplePytorchClassifier(3, multi_label=True)
        class_weights = torch.ones(3)
        loss = clf._get_default_criterion(class_weights, use_sample_weights=True)
        self.assertTrue(isinstance(loss, _LossAdapter2DTo1D))
        self.assertEqual('none', loss.reduction)

        self.assertTrue(isinstance(loss.base_loss_fct, BCEWithLogitsLoss))
        self.assertEqual('none', loss.base_loss_fct.reduction)

    def test_get_model_selection_default(self):
        model_selection = None
        model_selection = SimplePytorchClassifier._get_default_model_selection(model_selection)

        self.assertTrue(isinstance(model_selection, ModelSelection))
        self.assertEqual(4, len(model_selection.metrics))
        metric_names = set([metric.name for metric in model_selection.metrics])
        for metric_name in ['val_loss', 'val_acc', 'train_loss', 'train_acc']:
            self.assertTrue(metric_name in metric_names)

        self.assertEqual(1, len(model_selection._fields_config))
        self.assertTrue(model_selection._fields_config['early_stop'] == bool)

    def test_get_default_model_selection_none(self):
        model_selection = 'none'
        model_selection = SimplePytorchClassifier._get_default_model_selection(model_selection)
        self.assertTrue(isinstance(model_selection, NoopModelSelection))

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
