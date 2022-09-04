import unittest
import pytest

import numpy as np
from unittest.mock import patch

from unittest import mock
from unittest.mock import Mock
from scipy.sparse import issparse
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from small_text.training.early_stopping import EarlyStopping, NoopEarlyStopping
from small_text.training.metrics import Metric
from small_text.training.model_selection import ModelSelection, NoopModelSelection

try:
    import torch

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    from small_text.integrations.pytorch.classifiers.kimcnn import KimCNNClassifier
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


class _KimCNNClassifierTest(object):

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

    def test_fit_with_early_stopping(self):
        dataset = self._get_dataset(num_samples=20, num_classes=4)

        train_set = dataset[0:10]
        validation_set = dataset[10:]

        num_classes = 4
        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=self.multi_label,
                                      embedding_matrix=embedding_matrix,
                                      num_epochs=2)

        early_stopping = EarlyStopping(Metric('val_loss'))

        with mock.patch.object(early_stopping,
                               'check_early_stop',
                               wraps=early_stopping.check_early_stop) as check_early_stop_spy:

            classifier.fit(train_set, validation_set=validation_set, early_stopping=early_stopping)
            self.assertEqual(2, check_early_stop_spy.call_count)
            for i in range(2):
                self.assertEqual(i+1, check_early_stop_spy.call_args_list[i].args[0])
                self.assertTrue('train_acc' in check_early_stop_spy.call_args_list[i].args[1])

    def test_fit_with_early_stopping_default(self):
        dataset = self._get_dataset(num_samples=20, num_classes=4)

        train_set = dataset[0:10]
        validation_set = dataset[10:]

        num_classes = 4
        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=self.multi_label,
                                      embedding_matrix=embedding_matrix,
                                      num_epochs=2)

        with patch.object(classifier,
                          '_fit_main',
                          wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set)

            self.assertEqual(1, fit_main_spy.call_count)
            early_stopping_arg = fit_main_spy.call_args_list[0].args[3]
            self.assertTrue(isinstance(early_stopping_arg, EarlyStopping))
            self.assertEqual('val_loss', early_stopping_arg.metric.name)
            self.assertEqual(5, early_stopping_arg.patience)

    def test_fit_with_early_stopping_disabled(self):
        dataset = self._get_dataset(num_samples=20, num_classes=4)

        train_set = dataset[0:10]
        validation_set = dataset[10:]

        num_classes = 4
        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=self.multi_label,
                                      embedding_matrix=embedding_matrix,
                                      num_epochs=2)

        with patch.object(classifier,
                          '_fit_main',
                          wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set, early_stopping='none')

            self.assertEqual(1, fit_main_spy.call_count)
            self.assertTrue(isinstance(fit_main_spy.call_args_list[0].args[3], NoopEarlyStopping))

    def test_fit_with_model_selection_default(self):
        dataset = self._get_dataset(num_samples=20, num_classes=4)

        train_set = dataset[0:10]
        validation_set = dataset[10:]

        num_classes = 4
        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=self.multi_label,
                                      embedding_matrix=embedding_matrix,
                                      num_epochs=2)

        with patch.object(classifier,
                          '_fit_main',
                          wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set)

            self.assertEqual(1, fit_main_spy.call_count)
            model_selection_arg = fit_main_spy.call_args_list[0].args[4]
            self.assertTrue(isinstance(model_selection_arg, ModelSelection))

    def test_fit_with_model_selection_none(self):
        dataset = self._get_dataset(num_samples=20, num_classes=4)

        train_set = dataset[0:10]
        validation_set = dataset[10:]

        num_classes = 4
        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=self.multi_label,
                                      embedding_matrix=embedding_matrix,
                                      num_epochs=2)

        with patch.object(classifier,
                          '_fit_main',
                          wraps=classifier._fit_main) as fit_main_spy:
            classifier.fit(train_set, validation_set=validation_set, model_selection='none')

            self.assertEqual(1, fit_main_spy.call_count)
            model_selection_arg = fit_main_spy.call_args_list[0].args[4]
            self.assertTrue(isinstance(model_selection_arg, NoopModelSelection))

    def test_fit_with_optimizer_and_scheduler(self):
        ds = self._get_dataset()

        num_classes = 4
        embedding_matrix = torch.FloatTensor(np.random.rand(10, 100))
        classifier = KimCNNClassifier(num_classes,
                                      multi_label=self.multi_label,
                                      embedding_matrix=embedding_matrix)

        classifier.fit(ds)

        params = [param for param in classifier.model.parameters() if param.requires_grad]

        optimizer = AdamW(params, lr=5e-5)
        scheduler = LambdaLR(optimizer, lambda _: 1)

        with patch.object(classifier, '_train', wraps=classifier._train) as train_mock:
            classifier.fit(ds, optimizer=optimizer, scheduler=scheduler)
            train_mock.assert_called()

            call_args = train_mock.call_args[0]
            self.assertEqual(1, train_mock.call_count)

            self.assertEqual(optimizer, call_args[5])
            self.assertEqual(scheduler, call_args[6])


@pytest.mark.pytorch
class KimCNNClassifierSingleLabelTest(unittest.TestCase, _KimCNNClassifierTest):

    def setUp(self):
        self.multi_label = False


@pytest.mark.pytorch
class KimCNNClassifierMultiLabelTest(unittest.TestCase, _KimCNNClassifierTest):

    def setUp(self):
        self.multi_label = True
