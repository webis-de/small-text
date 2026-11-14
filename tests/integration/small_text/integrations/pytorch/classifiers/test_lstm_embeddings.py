import unittest

from unittest import mock

import numpy as np
import pytest

from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

try:
    import torch

    from small_text.integrations.pytorch.classifiers.base import AMPArguments
    from small_text.integrations.pytorch.classifiers.lstm import (
        LSTMClassifier,
        LSTMEmbeddingMixin,
    )
    from tests.utils.datasets import random_text_classification_dataset
except PytorchNotFoundError:
    pass


def default_module_selector(modules):
    return modules['fc']


class _LSTMEmbeddingTest(object):

    def _get_dataset(self, multi_label=False):
        return random_text_classification_dataset(12,
                                                  max_length=12,
                                                  num_classes=3,
                                                  multi_label=multi_label)

    def _get_classifier(self, **kwargs):
        return LSTMClassifier(3,
                              embedding_matrix=torch.Tensor(np.random.rand(10, 8)),
                              max_seq_len=12,
                              hidden_size=5,
                              num_epochs=1,
                              show_progress_bar=False,
                              **kwargs)

    def _embedding_kwargs(self):
        if self.embedding_method == LSTMEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
            return {'module_selector': default_module_selector}
        return {}

    def _assert_embedding_shape(self, classifier, embeddings, dataset):
        self.assertEqual(len(dataset), embeddings.shape[0])
        if self.embedding_method == LSTMEmbeddingMixin.EMBEDDING_METHOD_GRADIENT:
            expected_width = classifier.model.fc.weight.numel() * classifier.num_classes
        else:
            expected_width = classifier.model.fc.in_features
        self.assertEqual(expected_width, embeddings.shape[1])
        self.assertEqual(np.float32, embeddings.dtype)

    def test_embed_model_not_trained(self):
        classifier = self._get_classifier()

        with self.assertRaises(ValueError):
            classifier.embed(self._get_dataset(),
                             embedding_method=self.embedding_method,
                             **self._embedding_kwargs())

    def test_embed(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()
        classifier.fit(dataset)

        with mock.patch.object(classifier.model, 'eval', wraps=classifier.model.eval) as model_eval_spy:
            embeddings = classifier.embed(dataset,
                                          embedding_method=self.embedding_method,
                                          **self._embedding_kwargs())

        self.assertFalse(classifier.model.training)
        model_eval_spy.assert_called()
        self._assert_embedding_shape(classifier, embeddings, dataset)

    def test_embed_and_return_probabilities(self):
        dataset = self._get_dataset()
        classifier = self._get_classifier()
        classifier.fit(dataset)

        embeddings, probabilities = classifier.embed(dataset,
                                                      return_proba=True,
                                                      embedding_method=self.embedding_method,
                                                      **self._embedding_kwargs())

        self._assert_embedding_shape(classifier, embeddings, dataset)
        self.assertEqual(len(dataset), probabilities.shape[0])

    def test_embed_with_amp_configuration_on_cpu(self):
        dataset = self._get_dataset()
        amp_args = AMPArguments(use_amp=True, device_type='cpu', dtype=torch.bfloat16)
        classifier = self._get_classifier(amp_args=amp_args)
        classifier.fit(dataset)

        embeddings = classifier.embed(dataset,
                                      embedding_method=self.embedding_method,
                                      **self._embedding_kwargs())
        self._assert_embedding_shape(classifier, embeddings, dataset)


@pytest.mark.pytorch
class LSTMEmbeddingPooledTest(unittest.TestCase, _LSTMEmbeddingTest):

    def setUp(self):
        self.embedding_method = LSTMEmbeddingMixin.EMBEDDING_METHOD_POOLED


@pytest.mark.pytorch
class LSTMEmbeddingGradientTest(unittest.TestCase, _LSTMEmbeddingTest):

    def setUp(self):
        self.embedding_method = LSTMEmbeddingMixin.EMBEDDING_METHOD_GRADIENT


@pytest.mark.pytorch
class LSTMMultiLabelEmbeddingTest(unittest.TestCase):

    def test_gradient_embeddings_are_rejected_for_multi_label_classification(self):
        dataset = random_text_classification_dataset(12,
                                                     max_length=12,
                                                     num_classes=3,
                                                     multi_label=True)
        classifier = LSTMClassifier(3,
                                    multi_label=True,
                                    embedding_matrix=torch.Tensor(np.random.rand(10, 8)),
                                    max_seq_len=12,
                                    hidden_size=5,
                                    num_epochs=1,
                                    show_progress_bar=False)
        classifier.fit(dataset)

        with self.assertRaisesRegex(ValueError, 'only supported for single-label'):
            classifier.embed(dataset,
                             embedding_method=LSTMEmbeddingMixin.EMBEDDING_METHOD_GRADIENT,
                             module_selector=default_module_selector)
