import unittest
import pytest

import numpy as np
from unittest.mock import patch

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    from active_learning.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from active_learning.integrations.pytorch.datasets import PytorchTextClassificationDataset
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
class KimCNNInitTest(unittest.TestCase):

    def test_init_default_parameters(self):
        embedding_matrix = np.random.rand(10, 100)
        classifier = KimCNNClassifier(embedding_matrix=embedding_matrix)

        self.assertIsNotNone(classifier.device)
        self.assertEqual(10, classifier.num_epochs)
        self.assertEqual(25, classifier.mini_batch_size)
        self.assertIsNone(classifier.criterion)
        self.assertIsNone(classifier.optimizer)
        self.assertEqual(0.001, classifier.lr)
        self.assertEqual(60, classifier.max_seq_len)
        self.assertEqual(100, classifier.out_channels)
        self.assertEqual(0.5, classifier.dropout)
        self.assertEqual(0.1, classifier.validation_set_size)
        np.testing.assert_equal(embedding_matrix, classifier.embedding_matrix)
        self.assertEqual(0, classifier.padding_idx)
        self.assertEqual([3, 4, 5], classifier.kernel_heights)
        self.assertEqual(5, classifier.early_stopping)
        self.assertEqual(0.98, classifier.early_stopping_acc)
        self.assertIsNone(classifier.model)

    def test_init_parameters(self):

        device = 'cpu'
        num_epochs = 5
        mini_batch_size = 30
        max_seq_len = 100
        out_channels = 80
        dropout = 0.4
        validation_set_size = 0.15
        embedding_matrix = np.random.rand(5, 10)
        padding_idx = 1
        early_stopping = 10
        early_stopping_acc = 0.95

        classifier = KimCNNClassifier(device='cpu', num_epochs=num_epochs,
                                      mini_batch_size=mini_batch_size,
                                      max_seq_len=max_seq_len, out_channels=out_channels,
                                      dropout=dropout, validation_set_size=validation_set_size,
                                      embedding_matrix=embedding_matrix, padding_idx=padding_idx,
                                      early_stopping=early_stopping,
                                      early_stopping_acc=early_stopping_acc)

        self.assertEqual(device, classifier.device)
        self.assertEqual(num_epochs, classifier.num_epochs)
        self.assertEqual(mini_batch_size, classifier.mini_batch_size)
        self.assertIsNone(classifier.criterion)
        self.assertIsNone(classifier.optimizer)
        self.assertEqual(max_seq_len, classifier.max_seq_len)
        self.assertEqual(out_channels, classifier.out_channels)
        self.assertEqual(dropout, classifier.dropout)
        self.assertEqual(validation_set_size, classifier.validation_set_size)
        np.testing.assert_equal(embedding_matrix, classifier.embedding_matrix)
        self.assertEqual(padding_idx, classifier.padding_idx)
        self.assertEqual(early_stopping, classifier.early_stopping)
        self.assertEqual(early_stopping_acc, classifier.early_stopping_acc)
        self.assertIsNone(classifier.model)

    def test_init_without_embedding_matrix(self):
        num_epochs = 5
        mini_batch_size = 30
        max_seq_len = 100
        out_channels = 80
        dropout = 0.4
        validation_set_size = 0.15
        embedding_matrix = None
        padding_idx = 1
        early_stopping = 10
        early_stopping_acc = 0.95

        with self.assertRaises(ValueError):
            KimCNNClassifier(embedding_matrix=embedding_matrix, device='cpu',
                             num_epochs=num_epochs, mini_batch_size=mini_batch_size,
                             max_seq_len=max_seq_len, out_channels=out_channels,
                             dropout=dropout, validation_set_size=validation_set_size,
                             padding_idx=padding_idx, early_stopping=early_stopping,
                             early_stopping_acc=early_stopping_acc)

    @pytest.mark.skip(reason='should probably be removed')
    def test_fit_where_labels_is_none(self):

        dataset = random_text_classification_dataset(10)
        dataset.y = [None] * 10

        embedding_matrix = np.random.rand(5, 10)
        classifier = KimCNNClassifier(device='cpu', embedding_matrix=embedding_matrix)

        with self.assertRaises(ValueError):
            classifier.fit(dataset)

    def test_fit_without_validation_set(self):
        dataset = random_text_classification_dataset(10)

        embedding_matrix = np.random.rand(5, 10)
        classifier = KimCNNClassifier(device='cpu', embedding_matrix=embedding_matrix)

        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(dataset)
            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], PytorchTextClassificationDataset))
            self.assertTrue(isinstance(call_args[1], PytorchTextClassificationDataset))

            self.assertEqual(len(dataset), len(call_args[0]) + len(call_args[1]))

    def test_fit_with_validation_set(self):
        train = random_text_classification_dataset(8)
        valid = random_text_classification_dataset(2)

        embedding_matrix = np.random.rand(5, 10)
        classifier = KimCNNClassifier(device='cpu', embedding_matrix=embedding_matrix)

        with patch.object(classifier, '_fit_main') as fit_main_mock:
            classifier.fit(train, validation_set=valid)
            fit_main_mock.assert_called()

            call_args = fit_main_mock.call_args[0]
            self.assertTrue(isinstance(call_args[0], PytorchTextClassificationDataset))
            self.assertTrue(isinstance(call_args[1], PytorchTextClassificationDataset))

            self.assertEqual(len(train), len(call_args[0]))
            self.assertEqual(len(valid), len(call_args[1]))

    @pytest.mark.skip(reason='should probably be removed')
    def test_fit_with_validation_set_but_missing_labels(self):
        train = random_text_classification_dataset(8)
        valid = random_text_classification_dataset(2)
        valid.y = [None] * len(valid)

        embedding_matrix = np.random.rand(5, 10)
        classifier = KimCNNClassifier(device='cpu', embedding_matrix=embedding_matrix)

        with self.assertRaises(ValueError):
            classifier.fit(train, validation_set=valid)
