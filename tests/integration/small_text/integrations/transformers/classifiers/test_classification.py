import unittest
import pytest
import warnings

import numpy as np

from unittest import mock
from unittest.mock import patch, Mock

from parameterized import parameterized_class
from scipy.sparse import issparse

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import twenty_news_transformers
from tests.utils.testing import assert_array_not_equal

try:
    from small_text.integrations.transformers import TransformerModelArguments
    from small_text.integrations.transformers.classifiers import (
        TransformerBasedClassificationFactory,
        TransformerBasedClassification
    )
    from small_text.integrations.transformers.classifiers.classification import FineTuningArguments

    from tests.utils.datasets import random_transformer_dataset
except (ImportError, PytorchNotFoundError):
    # prevent "NameError: name 'TransformerBasedClassification' is not defined" in patch.object
    class TransformerBasedClassification(object):
        pass


@pytest.mark.skip
@pytest.mark.pytorch
@parameterized_class([{'embedding_method': 'avg', 'num_classes': 2},
                      {'embedding_method': 'cls', 'num_classes': 2},
                      {'embedding_method': 'avg', 'num_classes': 3},
                      {'embedding_method': 'cls', 'num_classes': 3}])
class EmbeddingTest(unittest.TestCase):

    def test_embed_model_not_fitted(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            'sshleifer/tiny-distilroberta-base',
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()

        with self.assertRaises(ValueError):
            clf.embed(train_set)

    def test_embed(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        with mock.patch.object(clf.model,
                               'eval',
                               wraps=clf.model.eval) as model_eval_spy:

            embeddings = clf.embed(train_set, embedding_method=self.embedding_method)
            model_eval_spy.assert_called()

        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(len(train_set), embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])

    def test_embed_with_layer_index(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        with mock.patch.object(clf.model,
                               'eval',
                               wraps=clf.model.eval) as model_eval_spy:
            embedding_one = clf.embed(train_set, embedding_method=self.embedding_method)
            model_eval_spy.assert_called()

        with mock.patch.object(clf.model,
                               'eval',
                               wraps=clf.model.eval) as model_eval_spy:
            embedding_two = clf.embed(train_set, embedding_method=self.embedding_method,
                                      hidden_layer_index=0)
            model_eval_spy.assert_called()

        assert_array_not_equal(embedding_one, embedding_two)

        self.assertEqual(2, len(embedding_one.shape))
        self.assertEqual(2, len(embedding_two.shape))
        self.assertEqual(len(train_set), embedding_one.shape[0])
        self.assertEqual(len(train_set), embedding_two.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embedding_one.shape[1])
        self.assertEqual(clf.model.config.hidden_size, embedding_two.shape[1])

    def test_embed_with_predictions(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        embeddings, predictions = clf.embed(train_set, return_proba=True, embedding_method=self.embedding_method)
        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(len(train_set), embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])
        self.assertEqual(len(train_set), predictions.shape[0])


class _TransformerBasedClassificationTest(object):

    def _get_dataset(self, num_samples=100, num_classes=4):
        return random_transformer_dataset(num_samples, max_length=60, num_classes=num_classes,
                                          multi_label=self.multi_label)

    @patch.object(TransformerBasedClassification, '_train')
    @patch.object(TransformerBasedClassification, '_select_best_model')
    def test_fit(self, select_best_model_mock, fake_train):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        clf.fit(train_set)

        # basically tests _get_layer_params for now

        fake_train.assert_called()
        select_best_model_mock.assert_called()

    def test_fit_with_class_weight(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        clf.fit(train_set)
        self.assertIsNotNone(clf.class_weights_)
        self.assertIsNotNone(clf.model)

    def test_fit_with_finetuning_args_and_scheduler_kwargs(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        finetuning_args = FineTuningArguments(5e-2, 0.99)
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1,
                                             fine_tuning_arguments=finetuning_args)

        train_set = self._get_dataset(num_samples=20)

        scheduler = Mock()

        with self.assertRaisesRegex(ValueError, 'When fine_tuning_arguments are provided'):
            clf.fit(train_set, scheduler=scheduler)

    def test_fit_with_scheduler_but_without_optimizer(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)

        scheduler = Mock()

        with self.assertRaisesRegex(ValueError, 'You must also pass an optimizer'):
            clf.fit(train_set, scheduler=scheduler)

    def test_fit_and_predict(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        test_set = self._get_dataset(num_samples=10)

        clf.fit(train_set)

        with mock.patch.object(clf.model, 'eval', wraps=clf.model.eval) as model_eval_spy, \
                mock.patch.object(clf.model, 'train', wraps=clf.model.train) as model_train_spy:

            y_pred = clf.predict(test_set)

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

    def test_fit_validate(self):

        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             class_weight='balanced',
                                             num_epochs=1)

        train_set = self._get_dataset(num_samples=20)
        valid_set = self._get_dataset(num_samples=5)

        clf.fit(train_set)

        with mock.patch.object(clf.model, 'eval', wraps=clf.model.eval) as model_eval_spy, \
             mock.patch.object(clf.model, 'train', wraps=clf.model.train) as model_train_spy:

            valid_loss, valid_acc = clf.validate(valid_set)

            model_eval_spy.assert_called()
            model_train_spy.assert_called_once_with(False)

        self.assertTrue(valid_loss >= 0)
        self.assertTrue(0.0 <= valid_acc <= 1.0)

    def test_validate_with_validations_per_epoch(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             num_epochs=1,
                                             mini_batch_size=10,
                                             validations_per_epoch=2)

        train_set = self._get_dataset(num_samples=20)

        with patch.object(clf, 'validate', wraps=clf.validate) as validate_spy:
            clf.fit(train_set)
            self.assertIsNotNone(clf.model)

            self.assertEqual(2, validate_spy.call_count)

    def test_validate_with_validations_per_epoch_too_large(self):
        model_args = TransformerModelArguments('sshleifer/tiny-distilroberta-base')
        clf = TransformerBasedClassification(model_args,
                                             4,
                                             multi_label=self.multi_label,
                                             num_epochs=1,
                                             mini_batch_size=20,
                                             validations_per_epoch=2)

        train_set = self._get_dataset(num_samples=20)

        with patch.object(clf, 'validate', wraps=clf.validate) as validate_spy, \
                warnings.catch_warnings(record=True) as w:
            clf.fit(train_set)
            self.assertIsNotNone(clf.model)

            self.assertEqual(1, validate_spy.call_count)

            expected_warning = 'validations_per_epoch=2 is greater than the maximum ' \
                               'possible batches of 1'
            found_warning = np.any([
                str(w_.message) == expected_warning and w_.category == RuntimeWarning
                for w_ in w])
            self.assertTrue(found_warning)


@pytest.mark.pytorch
class TransformerBasedClassificationSingleLabelTest(unittest.TestCase,
                                                    _TransformerBasedClassificationTest):

    def setUp(self):
        self.multi_label = False


@pytest.mark.pytorch
class TransformerBasedClassificationMultiLabelTest(unittest.TestCase,
                                                   _TransformerBasedClassificationTest):

    def setUp(self):
        self.multi_label = True
