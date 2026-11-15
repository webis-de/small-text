import unittest

import numpy as np
import pytest

from unittest.mock import Mock, patch

from small_text.base import LABEL_UNLABELED
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.classifiers.lstm import (
        LSTMClassifier,
        lstm_collate_fn,
    )
    from small_text.integrations.pytorch.datasets import PytorchDatasetView
    from small_text.integrations.pytorch.datasets import PytorchTextClassificationDataset

    from tests.utils.datasets import random_text_classification_dataset
except (ModuleNotFoundError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class LSTMClassifierTest(unittest.TestCase):

    def _get_clf(self, num_classes=2):
        embedding_matrix = torch.rand(10, 20)
        return LSTMClassifier(num_classes, embedding_matrix=embedding_matrix, num_epochs=2,
                              hidden_size=15, max_seq_len=20, device='cpu')

    def test_init_default_parameters(self):
        num_classes = 2
        embedding_matrix = np.random.rand(10, 100)
        classifier = LSTMClassifier(num_classes, embedding_matrix=embedding_matrix)

        self.assertEqual(num_classes, classifier.num_classes)
        self.assertFalse(classifier.multi_label)
        self.assertIsNotNone(classifier.device)
        self.assertEqual(10, classifier.num_epochs)
        self.assertEqual(25, classifier.train_batch_size)
        self.assertEqual(25, classifier.predict_batch_size)
        self.assertIsNone(classifier.criterion)
        self.assertIsNone(classifier.optimizer)
        self.assertIsNone(classifier.scheduler)
        self.assertEqual(0.001, classifier.lr)
        self.assertEqual(60, classifier.max_seq_len)
        self.assertEqual(100, classifier.hidden_size)
        self.assertEqual(0.5, classifier.dropout)
        self.assertEqual(0.1, classifier.validation_set_size)
        np.testing.assert_equal(embedding_matrix, classifier.embedding_matrix)
        self.assertEqual(0, classifier.padding_idx)
        self.assertFalse(classifier.bidirectional)
        self.assertFalse(classifier.freeze_embedding_layer)
        self.assertIsNone(classifier.class_weight)
        self.assertIsNotNone(classifier.amp_args)
        self.assertFalse(classifier.compile_model)
        self.assertIsNone(classifier.model)

    def test_init_parameters(self):
        num_classes = 2
        multi_label = True
        device = 'cpu'
        num_epochs = 5
        train_batch_size = 30
        predict_batch_size = 40
        max_seq_len = 100
        hidden_size = 80
        dropout = 0.4
        validation_set_size = 0.15
        embedding_matrix = np.random.rand(5, 10)
        padding_idx = 1
        bidirectional = True
        freeze_embedding_layer = True
        class_weight = 'balanced'
        amp_args = AMPArguments()
        compile_model = False

        classifier = LSTMClassifier(num_classes, multi_label=multi_label, device=device,
                                    num_epochs=num_epochs, train_batch_size=train_batch_size,
                                    predict_batch_size=predict_batch_size,
                                    max_seq_len=max_seq_len, hidden_size=hidden_size,
                                    dropout=dropout, validation_set_size=validation_set_size,
                                    embedding_matrix=embedding_matrix, padding_idx=padding_idx,
                                    bidirectional=bidirectional,
                                    freeze_embedding_layer=freeze_embedding_layer,
                                    class_weight=class_weight, amp_args=amp_args,
                                    compile_model=compile_model)

        self.assertEqual(num_classes, classifier.num_classes)
        self.assertEqual(multi_label, classifier.multi_label)
        self.assertEqual(device, classifier.device)
        self.assertEqual(num_epochs, classifier.num_epochs)
        self.assertEqual(train_batch_size, classifier.train_batch_size)
        self.assertEqual(predict_batch_size, classifier.predict_batch_size)
        self.assertIsNone(classifier.criterion)
        self.assertIsNone(classifier.optimizer)
        self.assertIsNone(classifier.scheduler)
        self.assertEqual(max_seq_len, classifier.max_seq_len)
        self.assertEqual(hidden_size, classifier.hidden_size)
        self.assertEqual(dropout, classifier.dropout)
        self.assertEqual(validation_set_size, classifier.validation_set_size)
        np.testing.assert_equal(embedding_matrix, classifier.embedding_matrix)
        self.assertEqual(padding_idx, classifier.padding_idx)
        self.assertTrue(classifier.bidirectional)
        self.assertTrue(classifier.freeze_embedding_layer)
        self.assertEqual(class_weight, classifier.class_weight)
        self.assertFalse(classifier.compile_model)
        self.assertIsNone(classifier.model)

    def test_init_without_embedding_matrix(self):
        with self.assertRaises(ValueError):
            LSTMClassifier(2, embedding_matrix=None, device='cpu')

    def test_collate_fn_pads_and_truncates_sequences(self):
        batch = [
            (torch.tensor([1, 2]), 0),
            (torch.tensor([3, 4, 5, 6]), 1),
        ]

        text, labels, weights = lstm_collate_fn(batch, max_seq_len=3, padding_idx=9)

        torch.testing.assert_close(text, torch.tensor([[1, 2, 9], [3, 4, 5]]))
        torch.testing.assert_close(labels, torch.tensor([0, 1]))
        torch.testing.assert_close(weights, torch.ones(2))

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
        train_set = random_text_classification_dataset(8, max_length=60,
                                                       num_classes=num_classes, multi_label=True)
        classifier = LSTMClassifier(num_classes,
                                    embedding_matrix=torch.rand(10, 100))

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set)

    def test_fit_with_train_set_mismatch_single_and_multi(self):
        num_classes = 3
        train_set = random_text_classification_dataset(8, max_length=60,
                                                       num_classes=num_classes)
        classifier = LSTMClassifier(num_classes, multi_label=True,
                                    embedding_matrix=torch.rand(10, 100))

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set)

    def test_fit_with_validation_set_mismatch_multi_and_single(self):
        num_classes = 3
        train_set = random_text_classification_dataset(15, max_length=60,
                                                       num_classes=num_classes)
        validation_set = random_text_classification_dataset(5, max_length=60,
                                                            num_classes=num_classes, multi_label=True)
        classifier = LSTMClassifier(num_classes,
                                    embedding_matrix=torch.rand(10, 100))

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_with_validation_set_mismatch_single_and_multi(self):
        num_classes = 3
        train_set = random_text_classification_dataset(15, max_length=60,
                                                       num_classes=num_classes, multi_label=True)
        validation_set = random_text_classification_dataset(5, max_length=60,
                                                            num_classes=num_classes)
        classifier = LSTMClassifier(num_classes, multi_label=True,
                                    embedding_matrix=torch.rand(10, 100))

        with self.assertRaisesRegex(ValueError, 'The classifier is configured for '):
            classifier.fit(train_set, validation_set=validation_set)

    def test_fit_where_y_train_contains_unlabeled(self):
        train_set = random_text_classification_dataset(10)
        train_set.y = np.array([0, 1] + [LABEL_UNLABELED] * 8)
        classifier = self._get_clf()

        with self.assertRaisesRegex(ValueError, 'Training set labels must be labeled'):
            classifier.fit(train_set)

    def test_fit_where_y_valid_contains_unlabeled(self):
        train_set = random_text_classification_dataset(8)
        validation_set = random_text_classification_dataset(8)
        validation_set.y = np.array([0, 1] + [LABEL_UNLABELED] * 6)
        classifier = self._get_clf()

        with self.assertRaisesRegex(ValueError, 'Validation set labels must be labeled'):
            classifier.fit(train_set, validation_set=validation_set)

    def test_returns_float32(self):
        train_set = random_text_classification_dataset(10)
        classifier = self._get_clf()
        classifier.fit(train_set)

        embeddings = classifier.embed(train_set)

        self.assertEqual(np.float32, embeddings.dtype)

    def test_predict_on_empty_data(self):
        train_set = random_text_classification_dataset(10)
        test_set = PytorchTextClassificationDataset(np.array([]), None)
        classifier = self._get_clf()
        classifier.fit(train_set)

        predictions = classifier.predict(test_set)

        self.assertEqual(0, predictions.shape[0])
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))

    def test_predict_proba_on_empty_data(self):
        train_set = random_text_classification_dataset(10)
        test_set = PytorchTextClassificationDataset(np.array([]), None)
        classifier = self._get_clf()
        classifier.fit(train_set)

        proba = classifier.predict_proba(test_set)

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

    def test_train_loss_is_weighted_by_batch_size(self):
        classifier = self._get_clf()

        class DataSet:
            data = [None] * 6

            def __len__(self):
                return len(self.data)

        batches = [
            (torch.empty(4, 1), torch.empty(4), torch.ones(4)),
            (torch.empty(2, 1), torch.empty(2), torch.ones(2)),
        ]
        optimizer = Mock()
        scheduler = Mock()
        scaler = Mock()
        amp_args = AMPArguments(use_amp=False, device_type='cpu', dtype=torch.bfloat16)

        with patch('small_text.integrations.pytorch.classifiers.lstm.dataloader', return_value=batches), \
             patch.object(classifier, '_train_single_batch',
                          side_effect=[(torch.tensor(1.0), torch.tensor(0.0)),
                                       (torch.tensor(3.0), torch.tensor(0.0))]):
            train_loss, _ = classifier._train_func(DataSet(), None, optimizer, scheduler, scaler, amp_args)

        self.assertAlmostEqual((1.0 * 4 + 3.0 * 2) / 6, train_loss)

    def test_validation_loss_is_weighted_by_batch_size(self):
        classifier = self._get_clf()
        classifier.model = Mock()
        classifier.model.device = torch.device('cpu')

        class DataSet:
            data = [None] * 6

            def __len__(self):
                return len(self.data)

        batches = [
            (torch.empty(4, 1), torch.empty(4), torch.ones(4)),
            (torch.empty(2, 1), torch.empty(2), torch.ones(2)),
        ]

        with patch('small_text.integrations.pytorch.classifiers.lstm.dataloader', return_value=batches), \
             patch.object(classifier, '_compute_loss',
                          side_effect=[torch.tensor(1.0), torch.tensor(3.0)]), \
             patch.object(classifier, 'sum_up_accuracy_', return_value=torch.tensor(0.0)):
            validation_loss, _ = classifier.validate(DataSet())

        self.assertAlmostEqual((1.0 * 4 + 3.0 * 2) / 6, validation_loss)
