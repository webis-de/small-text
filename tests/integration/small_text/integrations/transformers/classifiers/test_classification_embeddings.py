import unittest
import pytest
from unittest import mock

import torch

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError
from tests.utils.datasets import twenty_news_transformers
from tests.utils.testing import assert_array_not_equal

try:
    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.transformers import (
        TransformerBasedClassificationFactory,
        TransformerBasedEmbeddingMixin,
        TransformerModelArguments
    )
    from small_text.integrations.transformers.classifiers.classification import FineTuningArguments
except (ImportError, PytorchNotFoundError):
    pass


class _EmbeddingTest(object):

    def test_embed_model_not_fitted(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95)
        }
        clf_factory = TransformerBasedClassificationFactory(
            'sshleifer/tiny-distilroberta-base',
            self.num_classes,
            classification_kwargs=classifier_kwargs)

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
            classification_kwargs=classifier_kwargs)

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
            classification_kwargs=classifier_kwargs)

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

    def test_embed_with_proba(self):
        classifier_kwargs = {
            'fine_tuning_arguments': FineTuningArguments(0.2, 0.95),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            classification_kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        embeddings, proba = clf.embed(train_set,
                                      return_proba=True,
                                      embedding_method=self.embedding_method)
        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(len(train_set), embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])
        self.assertEqual(len(train_set), proba.shape[0])

    def test_embed_with_amp_args(self):
        classifier_kwargs = {
            'amp_args': AMPArguments(use_amp=True, device_type='cuda', dtype=torch.bfloat16),
            'num_epochs': 1
        }
        clf_factory = TransformerBasedClassificationFactory(
            TransformerModelArguments('sshleifer/tiny-distilroberta-base'),
            self.num_classes,
            classification_kwargs=classifier_kwargs)

        train_set = twenty_news_transformers(20, num_labels=self.num_classes)

        clf = clf_factory.new()
        clf.fit(train_set)

        embeddings, proba = clf.embed(train_set,
                                      return_proba=True,
                                      embedding_method=self.embedding_method)
        self.assertEqual(2, len(embeddings.shape))
        self.assertEqual(len(train_set), embeddings.shape[0])
        self.assertEqual(clf.model.config.hidden_size, embeddings.shape[1])
        self.assertEqual(len(train_set), proba.shape[0])


@pytest.mark.pytorch
class EmbeddingAvgBinaryClassificationTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.embedding_method = TransformerBasedEmbeddingMixin.EMBEDDING_METHOD_AVG
        self.num_classes = 2


@pytest.mark.pytorch
class EmbeddingAvgMultiClassClassificationTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.embedding_method = TransformerBasedEmbeddingMixin.EMBEDDING_METHOD_AVG
        self.num_classes = 3


@pytest.mark.pytorch
class EmbeddingClsTokenBinaryClassificationTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.embedding_method = TransformerBasedEmbeddingMixin.EMBEDDING_METHOD_CLS_TOKEN
        self.num_classes = 2


@pytest.mark.pytorch
class EmbeddingClsTokenMultiClassClassificationTest(unittest.TestCase, _EmbeddingTest):

    def setUp(self):
        self.embedding_method = TransformerBasedEmbeddingMixin.EMBEDDING_METHOD_CLS_TOKEN
        self.num_classes = 3
