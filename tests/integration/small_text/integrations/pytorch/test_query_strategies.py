import unittest

import pytest

import numpy as np

from unittest import mock
from small_text.integrations.pytorch.exceptions import PytorchNotFoundError

from tests.integration.small_text.query_strategies.test_query_strategies import (
    QueryStrategiesExhaustiveIntegrationTest
)

try:
    import torch
    from small_text.integrations.pytorch.classifiers import KimCNNFactory
    from small_text.integrations.pytorch.classifiers.factories import KimCNNClassifier
    from small_text.integrations.pytorch.query_strategies import ExpectedGradientLength, \
        ExpectedGradientLengthMaxWord, ExpectedGradientLengthLayer, BADGE

    from tests.utils.datasets import random_text_classification_dataset
except (ImportError, PytorchNotFoundError):
    pass


@pytest.mark.pytorch
class QueryStrategiesTest(QueryStrategiesExhaustiveIntegrationTest, unittest.TestCase):

    def _get_dataset(self, num_classes, multi_label=False):
        return random_text_classification_dataset(num_samples=60, max_length=10,
                                                  multi_label=multi_label, num_classes=num_classes)

    def _get_factory(self, num_classes, multi_label=False):

        return KimCNNFactory('kimcnn',
                             num_classes,
                             {'embedding_matrix': torch.rand(10, 20),
                              'num_epochs': 2})

    def test_expected_gradient_length(self):
        query_strategy = ExpectedGradientLength(2)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=2)

    def test_expected_gradient_length_max_word(self):
        query_strategy = ExpectedGradientLengthMaxWord(2, 'embedding')
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_expected_gradient_length_layer(self):
        query_strategy = ExpectedGradientLengthLayer(2, 'fc')
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_badge(self):
        query_strategy = BADGE(2)
        self._simple_exhaustive_active_learning_test(query_strategy, num_classes=2)

    def test_badge_multiclass(self):

        query_strategy = BADGE(6)
        self._simple_exhaustive_active_learning_test(query_strategy)

    def test_badge_with_classifier_that_does_not_return_embeddings_proba(self):
        # fake_embed return random embeddings and does not return probabilities
        def fake_embed(data_set, module_selector=lambda x: x['fc'], pbar='tqdm'):
            return np.random.rand(len(data_set), 10)

        query_strategy = BADGE(6)
        with mock.patch.object(KimCNNClassifier,
                               'embed',
                               wraps=fake_embed):

            self._simple_exhaustive_active_learning_test(query_strategy)


@pytest.mark.pytorch
class ExpectedGradientLengthMaxWordTest(unittest.TestCase):

    def test_query_with_layer_name_that_is_no_embedding_layer(self, num_samples=100):
        strategy = ExpectedGradientLengthMaxWord(4, 'fc', batch_size=100, device='cpu')

        embedding_matrix = torch.FloatTensor(np.random.rand(num_samples, 5))
        clf = KimCNNClassifier(2, embedding_matrix=embedding_matrix)
        dataset = random_text_classification_dataset(num_samples=num_samples)

        indices_labeled = np.random.choice(np.arange(num_samples), size=10, replace=False)
        indices_unlabeled = np.array([i for i in range(len(dataset))
                                      if i not in set(indices_labeled)])
        clf.fit(dataset[indices_labeled])

        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        with self.assertRaisesRegex(ValueError,
                                    r'Given parameter \(layer_name=fc\) is not'):
            strategy.query(clf, dataset, indices_labeled, indices_unlabeled, y)
