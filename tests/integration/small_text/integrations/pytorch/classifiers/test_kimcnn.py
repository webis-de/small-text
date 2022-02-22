import unittest
import pytest

import numpy as np

from unittest import mock
from unittest.mock import Mock
from parameterized import parameterized_class
from scipy.sparse import issparse
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch

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
