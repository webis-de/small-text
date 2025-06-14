import unittest
import pytest

import numpy as np
from unittest.mock import patch

from small_text.base import LABEL_UNLABELED
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from small_text.integrations.pytorch.datasets import PytorchDatasetView
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset

    from tests.utils.datasets import random_text_classification_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class KimCNNTest(unittest.TestCase):

    def _get_clf(self, num_classes=2):
        embedding_matrix = torch.rand(10, 20)
        return KimCNNClassifier(num_classes, embedding_matrix=embedding_matrix, num_epochs=2,
                                out_channels=15, max_seq_len=20, kernel_heights=[2, 3], device='cpu')

    def test_init_default_parameters(self):
        num_classes = 2
        embedding_matrix = np.random.rand(10, 100)
        classifier = KimCNNClassifier(num_classes, embedding_matrix=embedding_matrix)

        self.assertEqual(num_classes, classifier.num_classes)
        self.assertFalse(classifier.multi_label)
        self.assertIsNotNone(classifier.device)
        self.assertEqual(10, classifier.num_epochs)
        self.assertEqual(25, classifier.mini_batch_size)
        self.assertIsNone(classifier.criterion)
        self.assertIsNone(classifier.optimizer)
        self.assertIsNone(classifier.scheduler)
        self.assertEqual(0.001, classifier.lr)
        self.assertEqual(60, classifier.max_seq_len)
        self.assertEqual(100, classifier.out_channels)
        self.assertEqual(0, classifier.filter_padding)
        self.assertEqual(0.5, classifier.dropout)
        self.assertEqual(0.1, classifier.validation_set_size)
        np.testing.assert_equal(embedding_matrix, classifier.embedding_matrix)
        self.assertEqual(0, classifier.padding_idx)
        self.assertEqual([3, 4, 5], classifier.kernel_heights)
        self.assertIsNone(classifier.class_weight)
        self.assertIsNotNone(classifier.amp_args)
        self.assertFalse(classifier.compile_model)
        self.assertIsNone(classifier.model)

    def test_init_parameters(self):
        num_classes = 2
        multi_label = True
        device = 'cpu'
        num_epochs = 5
        mini_batch_size = 30
        max_seq_len = 100
        out_channels = 80
        filter_padding = 0
        dropout = 0.4
        validation_set_size = 0.15
        embedding_matrix = np.random.rand(5, 10)
        padding_idx = 1
        class_weight = 'balanced'
        amp_args = AMPArguments()
        compile_model = False

        classifier = KimCNNClassifier(num_classes, multi_label=multi_label, device='cpu',
                                      num_epochs=num_epochs, mini_batch_size=mini_batch_size,
                                      max_seq_len=max_seq_len, out_channels=out_channels,
                                      filter_padding=filter_padding, dropout=dropout,
                                      validation_set_size=validation_set_size,
                                      embedding_matrix=embedding_matrix, padding_idx=padding_idx,
                                      class_weight=class_weight, amp_args=amp_args,
                                      compile_model=compile_model)

        self.assertEqual(num_classes, classifier.num_classes)
        self.assertEqual(multi_label, classifier.multi_label)
        self.assertEqual(device, classifier.device)
        self.assertEqual(num_epochs, classifier.num_epochs)
        self.assertEqual(mini_batch_size, classifier.mini_batch_size)
        self.assertIsNone(classifier.criterion)
        self.assertIsNone(classifier.optimizer)
        self.assertIsNone(classifier.scheduler)
        self.assertEqual(max_seq_len, classifier.max_seq_len)
        self.assertEqual(out_channels, classifier.out_channels)
        self.assertEqual(filter_padding, classifier.filter_padding)
        self.assertEqual(dropout, classifier.dropout)
        self.assertEqual(validation_set_size, classifier.validation_set_size)
        np.testing.assert_equal(embedding_matrix, classifier.embedding_matrix)
        self.assertEqual(padding_idx, classifier.padding_idx)
        self.assertEqual(class_weight, classifier.class_weight)
        # TODO: reenable when amp args is comparable
        # self.assertEqual(amp_args, classifier.amp_args)
        self.assertFalse(classifier.compile_model)
        self.assertIsNone(classifier.model)

    def test_init_without_embedding_matrix(self):
        num_classes = 2
        num_epochs = 5
        mini_batch_size = 30
        max_seq_len = 100
        out_channels = 80
        dropout = 0.4
        validation_set_size = 0.15
        embedding_matrix = None
        padding_idx = 1

        with self.assertRaises(ValueError):
            KimCNNClassifier(num_classes, embedding_matrix=embedding_matrix, device='cpu',
                             num_epochs=num_epochs, mini_batch_size=mini_batch_size,
                             max_seq_len=max_seq_len, out_channels=out_channels,
                             dropout=dropout, validation_set_size=validation_set_size,
                             padding_idx=padding_idx)

    def test_fit_without_validation_set(self):
        dataset = random_text_classification_dataset(10)
        classifier = self._get_clf()

        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(dataset)
            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], PytorchDatasetView))
            self.assertTrue(isinstance(call_args[1], PytorchDatasetView))

            self.assertEqual(len(dataset), len(call_args[0]) + len(call_args[1]))

    def test_fit_with_validation_set(self):
        train = random_text_classification_dataset(8)
        valid = random_text_classification_dataset(2)

        classifier = self._get_clf()

        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(train, validation_set=valid)
            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], PytorchTextClassificationDataset))
            self.assertTrue(isinstance(call_args[1], PytorchTextClassificationDataset))

            self.assertEqual(len(train), len(call_args[0]))
            self.assertEqual(len(valid), len(call_args[1]))

    def test_fit_with_train_set_mismatch_multi_and_single(self):
        num_classes = 3
        train_set = random_text_classification_dataset(8,
                                                       max_length=60,
                                                       num_classes=num_classes,
                                                       multi_label=True)

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      embedding_matrix=embedding_matrix)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set)

    def test_fit_with_train_set_mismatch_single_and_multi(self):
        num_classes = 3
        train_set = random_text_classification_dataset(8,
                                                       max_length=60,
                                                       num_classes=num_classes)

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=True,
                                      embedding_matrix=embedding_matrix)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set)

    def test_fit_with_validation_set_mismatch_multi_and_single(self):
        num_classes = 3
        train_set = random_text_classification_dataset(15,
                                                       max_length=60,
                                                       num_classes=num_classes)
        validation_set = random_text_classification_dataset(5,
                                                            max_length=60,
                                                            num_classes=num_classes,
                                                            multi_label=True)

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      embedding_matrix=embedding_matrix)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_with_validation_set_mismatch_single_and_multi(self):
        num_classes = 3
        train_set = random_text_classification_dataset(15,
                                                       max_length=60,
                                                       num_classes=num_classes,
                                                       multi_label=True)
        validation_set = random_text_classification_dataset(5,
                                                            max_length=60,
                                                            num_classes=num_classes)

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=True,
                                      embedding_matrix=embedding_matrix)

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_where_y_train_contains_unlabeled(self):
        train_set = random_text_classification_dataset(10)
        train_set.y = np.array([LABEL_UNLABELED] * 10)

        classifier = self._get_clf()

        with self.assertRaisesRegex(ValueError, 'Training set labels must be labeled'):
            classifier.fit(train_set)

    def test_fit_where_y_valid_contains_unlabeled(self):
        train_set = random_text_classification_dataset(8)
        validation_set = random_text_classification_dataset(8)
        validation_set.y = np.array([LABEL_UNLABELED] * 8)

        classifier = self._get_clf()

        with self.assertRaisesRegex(ValueError, 'Validation set labels must be labeled'):
            classifier.fit(train_set, validation_set=validation_set)

    def test_predict_on_empty_data(self):
        train_set = random_text_classification_dataset(10)
        test_set = PytorchTextClassificationDataset(np.array([]), None)

        clf = self._get_clf()
        clf.fit(train_set)

        predictions = clf.predict(test_set)
        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        train_set = random_text_classification_dataset(10)
        test_set = PytorchTextClassificationDataset(np.array([]), None)

        clf = self._get_clf()
        clf.fit(train_set)

        proba = clf.predict_proba(test_set)
        self.assertEqual(0, proba.shape[0])
        self.assertTrue(np.issubdtype(proba.dtype, float))

    def test_fit_with_invalid_sample_weights(self):
        train_set = random_text_classification_dataset(8)
        validation_set = random_text_classification_dataset(8)

        classifier = self._get_clf()

        weights = np.random.randn(len(train_set))
        weights[0] = -1

        with self.assertRaisesRegex(ValueError, 'Weights must be greater zero.'):
            classifier.fit(train_set, validation_set=validation_set, weights=weights)
