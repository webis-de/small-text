import unittest
import pytest

from unittest.mock import patch

from parameterized import parameterized_class

from active_learning.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import twenty_news_transformers

try:
    import torch
    from active_learning.integrations.transformers import TransformerModelArguments
    from active_learning.integrations.transformers.classifiers import TransformerBasedClassificationFactory, TransformerBasedClassification
    from active_learning.integrations.transformers.classifiers.classification import FineTuningArguments, _get_layer_params
    from active_learning.integrations.transformers.datasets import TransformersDataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
@parameterized_class([{'num_classes': 2}, {'num_classes': 2}])
class EmbeddingTest(unittest.TestCase):

    def test_embed_model_not_fitted(self):
        classifier_kwargs = {
            'num_classes': self.num_classes, 'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            'sshleifer/tiny-distilroberta-base', kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()

        with self.assertRaises(ValueError):
            clf.embed(train_set)

    def test_embed_avg(self):
        classifier_kwargs = {
            'num_classes': self.num_classes, 'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        embeddings = clf.embed(train_set)
        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(20, embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])

    def test_embed_pooled(self):
        classifier_kwargs = {
            'num_classes': self.num_classes, 'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        embeddings = clf.embed(train_set, embedding_method='pooled')
        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(20, embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])


@pytest.mark.pytorch
class ClassificationTest(unittest.TestCase):

    @patch.object(TransformerBasedClassification, '_train')
    @patch.object(TransformerBasedClassification, '_select_best_model')
    def test_fit_distilroberta(self, select_best_model_mock, fake_train):
        classifier_kwargs = {
            'num_classes': 2, 'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            kwargs=classifier_kwargs)

        x = twenty_news_transformers(20, num_labels=2)

        clf = clf_factory.new()
        clf.fit(x)

        # basically tests _get_layer_params for now

        fake_train.assert_called()
        select_best_model_mock.assert_called()

    @patch.object(TransformerBasedClassification, '_train')
    @patch.object(TransformerBasedClassification, '_select_best_model')
    def test_test_fit_distilbert(self, select_best_model_mock, fake_train):
        classifier_kwargs = {
            'num_classes': 2, 'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('distilbert-base-cased'),
            kwargs=classifier_kwargs)

        x = twenty_news_transformers(20, num_labels=2)

        clf = clf_factory.new()
        clf.fit(x)

        # basically tests _get_layer_params for now

        fake_train.assert_called()
        select_best_model_mock.assert_called()
