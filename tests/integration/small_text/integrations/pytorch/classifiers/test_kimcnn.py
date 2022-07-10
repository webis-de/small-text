import unittest
import pytest

import numpy as np
from unittest.mock import patch

from unittest import mock
from unittest.mock import Mock
from parameterized import parameterized_class
from scipy.sparse import issparse
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


@pytest.mark.pytorch
@parameterized_class([{'multi_label': True},
                      {'multi_label': False}])
class KimCNNClassifierTest(unittest.TestCase):

    def _get_dataset(self, num_samples=100, num_classes=4):
        return random_text_classification_dataset(num_samples, max_length=60, num_classes=num_classes,
                                                  multi_label=self.multi_label)

    def test_fit_with_scheduler_but_without_optimizer(self):
        ds = self._get_dataset()

        num_classes = 4

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        clf = KimCNNClassifier(num_classes,
                               multi_label=self.multi_label,
                               embedding_matrix=embedding_matrix)

        scheduler = Mock()

        with self.assertRaisesRegex(ValueError, 'You must also pass an optimizer'):
            clf.fit(ds, scheduler=scheduler)

    def test_fit_and_predict(self):
        ds = self._get_dataset()

        num_classes = 4

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        clf = KimCNNClassifier(num_classes,
                               multi_label=self.multi_label,
                               embedding_matrix=embedding_matrix)
        clf.fit(ds)

        with mock.patch.object(clf.model, 'eval', wraps=clf.model.eval) as model_eval_spy, \
             mock.patch.object(clf.model, 'train', wraps=clf.model.train) as model_train_spy:

            y_pred = clf.predict(ds)
            self.assertEqual(len(ds), y_pred.shape[0])

            model_eval_spy.assert_called()
            model_train_spy.assert_called_once_with(False)

        if self.multi_label:
            self.assertTrue(issparse(y_pred))
            self.assertEqual(y_pred.dtype, np.int64)
            self.assertTrue(np.logical_or(y_pred.indices.all() >= 0, y_pred.indices.all() <= 3))
        else:
            self.assertTrue(isinstance(y_pred, np.ndarray))
            self.assertTrue(np.all([isinstance(y, np.int64) for y in y_pred]))
            self.assertTrue(np.logical_or(y_pred.all() >= 0, y_pred.all() <= 3))

    def test_fit_and_predict_with_sample_weights(self):
        ds = self._get_dataset()

        num_classes = 4

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        clf = KimCNNClassifier(num_classes,
                               multi_label=self.multi_label,
                               embedding_matrix=embedding_matrix)

        weights = np.random.randn(len(ds))
        weights = weights - weights.min() + 1e-8

        clf.fit(ds, weights=weights)

        with mock.patch.object(clf.model, 'eval', wraps=clf.model.eval) as model_eval_spy, \
             mock.patch.object(clf.model, 'train', wraps=clf.model.train) as model_train_spy:

            y_pred = clf.predict(ds)
            self.assertEqual(len(ds), y_pred.shape[0])

            model_eval_spy.assert_called()
            model_train_spy.assert_called_once_with(False)

        if self.multi_label:
            self.assertTrue(issparse(y_pred))
            self.assertEqual(y_pred.dtype, np.int64)
            self.assertTrue(np.logical_or(y_pred.indices.all() >= 0, y_pred.indices.all() <= 3))
        else:
            self.assertTrue(isinstance(y_pred, np.ndarray))
            self.assertTrue(np.all([isinstance(y, np.int64) for y in y_pred]))
            self.assertTrue(np.logical_or(y_pred.all() >= 0, y_pred.all() <= 3))

    def test_fit_and_validate(self):
        ds = self._get_dataset()

        num_classes = 4

        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        clf = KimCNNClassifier(num_classes,
                               multi_label=self.multi_label,
                               embedding_matrix=embedding_matrix)
        clf.fit(ds)

        with mock.patch.object(clf.model, 'eval', wraps=clf.model.eval) as model_eval_spy, \
             mock.patch.object(clf.model, 'train', wraps=clf.model.train) as model_train_spy:

            loss, acc = clf.validate(ds)
            self.assertTrue(loss >= 0)
            self.assertTrue(0 <= acc <= 1)

            model_eval_spy.assert_called()
            model_train_spy.assert_called_once_with(False)

    def test_fit_with_optimizer_and_scheduler(self):
        ds = self._get_dataset()

        num_classes = 4
        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=self.multi_label,
                                      embedding_matrix=embedding_matrix)

        classifier.fit(ds)

        params = [param for param in classifier.model.parameters()
                  if param.requires_grad]

        optimizer = AdamW(params, lr=5e-5)
        steps = (len(ds) // classifier.mini_batch_size) \
            + int(len(ds) % classifier.mini_batch_size != 0)

        scheduler = get_linear_schedule_with_warmup(optimizer, 0.1 * steps, steps)

        with patch.object(classifier, '_train', wraps=classifier._train) as train_mock:
            classifier.fit(ds, optimizer=optimizer, scheduler=scheduler)
            train_mock.assert_called()

            call_args = train_mock.call_args[0]
            self.assertEqual(1, train_mock.call_count)

            self.assertEqual(optimizer, call_args[4])
            self.assertEqual(scheduler, call_args[5])
